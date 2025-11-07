"""
Outlier Management System for Shipment Master Taxonomy

This script fetches products from product_master, retrieves corresponding shipment data,
performs rolling IQR outlier detection, and updates outlier flags in BigQuery.
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'outlier_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OutlierAnalyzer:
    """
    Analyzes shipment data for price outliers using rolling IQR method.
    """
    
    def __init__(
        self,
        project_id: str = "dev-tradyon-data",
        dataset_id: str = "tradyon",
        span: int = 600,
        mad_multiplier: float = 3.0,
        median_pct_window: float = 0.30,
        value_threshold: float = 2500.0,
        weight_threshold: float = 500.0,
        max_unit_price: Optional[float] = None
    ):
        """
        Initialize the outlier analyzer.
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            span: EWM span for smoothness (larger = smoother, more consistent)
            mad_multiplier: Multiplier for MAD-based bounds (default: 3.0)
            median_pct_window: Percentage window for median baseline (default: 0.30 = ±30%)
            value_threshold: Minimum value_of_goods_usd to include
            weight_threshold: Minimum weight_in_kg to include
            max_unit_price: Maximum unit_price to include (optional)
        """
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.product_master_table = f"{project_id}.{dataset_id}.product_master"
        self.shipment_taxonomy_table = f"{project_id}.{dataset_id}.shipment_master_taxonomy"
        
        # Analysis parameters (EWM-MAD method)
        self.span = span
        self.mad_multiplier = mad_multiplier
        self.median_pct_window = median_pct_window
        self.value_threshold = value_threshold
        self.weight_threshold = weight_threshold
        self.max_unit_price = max_unit_price
        
        logger.info(f"Initialized OutlierAnalyzer with span={span}, "
                   f"MAD multiplier={mad_multiplier}, median % window=±{median_pct_window:.1%}")
    
    def get_all_products(self) -> pd.DataFrame:
        """
        Fetch all products from product_master table.
        
        Returns:
            DataFrame with product information
        """
        logger.info("Fetching products from product_master...")
        
        query = f"""
        SELECT 
            product_id,
            product_name,
            synonymns,
            product_schema
        FROM `{self.product_master_table}`
        """
        
        df = self.client.query(query).to_dataframe()
        logger.info(f"Fetched {len(df)} products from product_master")
        
        return df
    
    def get_shipments_for_product(self, product_name: str) -> pd.DataFrame:
        """
        Fetch shipment data for a specific product from shipment_master_taxonomy.
        
        Args:
            product_name: Name of the product to fetch shipments for
            
        Returns:
            DataFrame with shipment data
        """
        logger.info(f"Fetching shipments for product: {product_name}")
        
        query = f"""
        SELECT 
            shipment_id,
            date,
            hs_code,
            goods_shipped,
            shipment_destination,
            shipment_origin,
            value_of_goods_usd,
            weight_in_kg,
            unit_price,
            product_name,
            attributes,
            is_multi_product_shipment,
            shipper_business_id,
            is_outlier,
            outlier_type
        FROM `{self.shipment_taxonomy_table}`
        WHERE product_name = @product_name
            AND is_multi_product_shipment = FALSE
            AND weight_in_kg > 0
            AND unit_price > 0
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("product_name", "STRING", product_name)
            ]
        )
        
        df = self.client.query(query, job_config=job_config).to_dataframe()
        logger.info(f"Fetched {len(df)} shipments for {product_name}")
        
        return df
    
    def apply_filters(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply pre-filtering to remove extreme values before outlier analysis.
        Also marks filtered records as outliers.
        
        Args:
            df: Input DataFrame with shipment data
            
        Returns:
            Tuple of (filtered DataFrame for analysis, filtered-out records marked as outliers)
        """
        if df.empty:
            return df, pd.DataFrame()
        
        before_rows = len(df)
        
        # Apply value and weight thresholds
        mask_value = df['value_of_goods_usd'] >= self.value_threshold
        mask_weight = df['weight_in_kg'] >= self.weight_threshold
        mask = mask_value & mask_weight
        
        df_filtered = df[mask].copy()
        df_excluded = df[~mask].copy()
        
        # Mark excluded records as outliers with specific types
        if not df_excluded.empty:
            df_excluded['is_outlier'] = True
            # Determine outlier type based on which threshold failed
            conditions = [
                (~mask_value) & (~mask_weight),  # Both failed
                ~mask_value,                      # Only value failed
                ~mask_weight                      # Only weight failed
            ]
            choices = ['LOW_VALUE_LOW_WEIGHT', 'LOW_VALUE', 'LOW_WEIGHT']
            df_excluded['outlier_type'] = pd.Series(
                [choices[next(i for i, c in enumerate(conditions) if c.loc[idx])] 
                 for idx in df_excluded.index],
                index=df_excluded.index
            )
        
        dropped = before_rows - len(df_filtered)
        logger.info(f"Applied filters: value >= {self.value_threshold}, weight >= {self.weight_threshold}")
        logger.info(f"Rows before: {before_rows}, after: {len(df_filtered)}, dropped: {dropped} "
                   f"({dropped / before_rows * 100:.2f}%)")
        
        # Apply max unit price if specified
        if self.max_unit_price is not None and not df_filtered.empty:
            before_price = len(df_filtered)
            price_mask = (df_filtered['unit_price'] <= self.max_unit_price) | \
                        df_filtered['unit_price'].isna()
            df_price_excluded = df_filtered[~price_mask].copy()
            
            # Mark price-excluded records as outliers
            if not df_price_excluded.empty:
                df_price_excluded['is_outlier'] = True
                df_price_excluded['outlier_type'] = 'HIGH_PRICE'
                df_excluded = pd.concat([df_excluded, df_price_excluded], ignore_index=True)
            
            df_filtered = df_filtered[price_mask].copy()
            dropped_price = before_price - len(df_filtered)
            logger.info(f"Applied price filter: unit_price <= {self.max_unit_price}")
            logger.info(f"Dropped {dropped_price} rows ({dropped_price / before_price * 100:.2f}%)")
        
        return df_filtered, df_excluded
    
    def _find_best_category_column(self, df: pd.DataFrame) -> Tuple[str, str]:
        """
        Dynamically find the best attribute to use for category segmentation.
        
        Looks at the 'attributes' JSON column and finds the attribute with:
        1. Most non-null values
        2. Reasonable number of unique categories (2-20)
        
        Args:
            df: Input DataFrame with 'attributes' column
            
        Returns:
            Tuple of (column_name, category_type) e.g., ('category', 'physical_form')
        """
        import json
        
        if 'attributes' not in df.columns:
            logger.warning("No 'attributes' column found, will use 'Uncategorized'")
            return 'auto_category', 'none'
        
        # Collect all possible attribute keys and their statistics
        attribute_stats = {}
        
        for idx, attrs in df['attributes'].items():
            if pd.isna(attrs):
                continue
                
            # Parse if string
            if isinstance(attrs, str):
                try:
                    attrs = json.loads(attrs)
                except:
                    continue
            
            if not isinstance(attrs, dict):
                continue
            
            # Count each attribute key
            for key, value in attrs.items():
                if value and str(value).strip() and str(value).lower() not in ['none', 'null', 'unknown', '']:
                    if key not in attribute_stats:
                        attribute_stats[key] = {'count': 0, 'unique_values': set()}
                    attribute_stats[key]['count'] += 1
                    attribute_stats[key]['unique_values'].add(str(value))
        
        if not attribute_stats:
            logger.warning("No valid attributes found in 'attributes' column")
            return 'auto_category', 'none'
        
        # Score each attribute
        # Prefer: high non-null count, reasonable number of unique values (2-20)
        best_attribute = None
        best_score = -1
        
        for attr_name, stats in attribute_stats.items():
            non_null_count = stats['count']
            unique_count = len(stats['unique_values'])
            
            # Skip if too few unique values (not useful) or too many (too fragmented)
            if unique_count < 2 or unique_count > 20:
                continue
            
            # Score = non-null percentage * ideal_category_bonus
            non_null_pct = non_null_count / len(df)
            
            # Bonus for having 2-10 categories (sweet spot)
            if 2 <= unique_count <= 10:
                category_bonus = 1.2
            else:
                category_bonus = 1.0
            
            score = non_null_pct * category_bonus
            
            logger.info(f"  Attribute '{attr_name}': {non_null_count}/{len(df)} non-null "
                       f"({non_null_pct:.1%}), {unique_count} unique values, score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_attribute = attr_name
        
        if best_attribute:
            logger.info(f"✓ Selected '{best_attribute}' for category segmentation (score: {best_score:.3f})")
            return 'auto_category', best_attribute
        else:
            logger.warning("No suitable attribute found for segmentation, using single group")
            return 'auto_category', 'none'
    
    def detect_outliers_by_category(
        self,
        df: pd.DataFrame,
        category_col: str = 'auto',
        span: int = 600,
        mad_multiplier: float = 3.0,
        median_pct_window: float = 0.30
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform EWM-MAD outlier detection grouped by category.
        
        Uses exponentially weighted moving median and MAD on log-transformed prices
        to produce stable decision bands across time.
        
        Args:
            df: Input DataFrame with shipment data
            category_col: Column name to group by. Use 'auto' to automatically detect
                         the best attribute from the 'attributes' JSON column
            span: EWM span for smoothness (larger = smoother, more consistent)
            mad_multiplier: Multiplier for MAD-based bounds
            median_pct_window: Percentage window for median baseline stabilization
            
        Returns:
            Tuple of (updated DataFrame with outlier flags, results dictionary)
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for outlier detection")
            return df, {}
        
        # Prepare DataFrame
        work_df = df.copy()
        work_df['date'] = pd.to_datetime(work_df['date'], errors='coerce')
        
        # Auto-detect best category column if requested
        if category_col == 'auto':
            logger.info("Auto-detecting best category attribute...")
            _, selected_attr = self._find_best_category_column(work_df)
            category_col = 'auto_category'
            
            # Extract the selected attribute
            import json
            def extract_selected_attr(attrs):
                if pd.isna(attrs):
                    return 'All'
                if isinstance(attrs, str):
                    try:
                        attrs = json.loads(attrs)
                    except:
                        return 'All'
                if isinstance(attrs, dict):
                    val = attrs.get(selected_attr)
                    if val and str(val).strip():
                        return str(val)
                return 'All'
            
            work_df[category_col] = work_df['attributes'].apply(extract_selected_attr)
        
        # Extract category from attributes JSON if not already a column
        elif category_col not in work_df.columns and 'attributes' in work_df.columns:
            import json
            def extract_category(attrs):
                if pd.isna(attrs) or not isinstance(attrs, dict):
                    return 'Uncategorized'
                # Try multiple possible keys
                for key in [category_col, 'Category', 'physical_form', 'Physical_Form']:
                    val = attrs.get(key)
                    if val:
                        return val
                return 'Uncategorized'
            
            work_df[category_col] = work_df['attributes'].apply(extract_category)
        elif category_col not in work_df.columns:
            work_df[category_col] = 'Uncategorized'
        
        # Fill missing categories
        work_df[category_col] = work_df[category_col].fillna('All')
        
        logger.info("="*80)
        logger.info(f"EWM-MAD OUTLIER DETECTION (Grouped by {category_col})")
        logger.info(f"EWM span: {span}, MAD multiplier: {mad_multiplier}, "
                   f"Median pct window: ±{median_pct_window:.1%}")
        logger.info("="*80)
        
        updated_dfs = []
        outlier_results = {}
        
        for cat, df_cat in work_df.groupby(category_col):
            logger.info(f"\nAnalyzing Category: {cat}")
            
            df_cat = df_cat.reset_index().rename(columns={'index': '_orig_idx'})
            df_clean = (df_cat
                       .dropna(subset=['unit_price'])
                       .sort_values('date'))
            
            # Minimum data requirement
            min_required = 50
            if len(df_clean) < min_required:
                logger.warning(f"  Insufficient rows ({len(df_clean)}) < {min_required}")
                # Mark all as non-outliers
                df_cat['is_outlier'] = False
                df_cat['outlier_type'] = None
                updated_dfs.append(df_cat.drop(columns=['_orig_idx']))
                
                outlier_results[cat] = {
                    'total_shipments': len(df_cat),
                    'analyzed_shipments': len(df_clean),
                    'outlier_shipments': 0,
                    'lower_outlier_shipments': 0,
                    'upper_outlier_shipments': 0,
                    'outlier_percentage': 0.0
                }
                continue
            
            # Log-transform for stability
            df_clean['log_price'] = np.log(df_clean['unit_price'].clip(lower=1e-6))
            
            # Exponentially weighted median approximation: use EWM mean as proxy
            ewm_mean = df_clean['log_price'].ewm(span=span, min_periods=10).mean()
            ewm_mad = (df_clean['log_price'] - ewm_mean).abs().ewm(span=span, min_periods=10).mean()
            
            # Basic MAD-based bounds
            df_clean['lower_bound_mad'] = np.exp(ewm_mean - mad_multiplier * ewm_mad)
            df_clean['upper_bound_mad'] = np.exp(ewm_mean + mad_multiplier * ewm_mad)
            
            # Median % window stabilization
            median_baseline = np.exp(ewm_mean)
            delta = median_baseline * median_pct_window
            df_clean['lower_bound_pct'] = median_baseline - delta
            df_clean['upper_bound_pct'] = median_baseline + delta
            
            # Final bounds: use the more conservative of the two methods
            df_clean['lower_bound'] = pd.concat(
                [df_clean['lower_bound_mad'], df_clean['lower_bound_pct']], axis=1
            ).min(axis=1, skipna=True)
            df_clean['upper_bound'] = pd.concat(
                [df_clean['upper_bound_mad'], df_clean['upper_bound_pct']], axis=1
            ).max(axis=1, skipna=True)
            df_clean['trend_line'] = np.exp(ewm_mean)
            
            # Detect outliers
            df_clean['is_lower_outlier'] = df_clean['unit_price'] < df_clean['lower_bound']
            df_clean['is_upper_outlier'] = df_clean['unit_price'] > df_clean['upper_bound']
            df_clean['is_outlier'] = df_clean['is_lower_outlier'] | df_clean['is_upper_outlier']
            
            # Set outlier type
            df_clean['outlier_type'] = None
            df_clean.loc[df_clean['is_lower_outlier'], 'outlier_type'] = 'LOWER_PRICE'
            df_clean.loc[df_clean['is_upper_outlier'], 'outlier_type'] = 'UPPER_PRICE'
            
            # Merge back to original category DataFrame
            merge_cols = ['_orig_idx', 'is_outlier', 'outlier_type']
            df_cat = df_cat.merge(df_clean[merge_cols], on='_orig_idx', how='left', suffixes=('', '_new'))
            
            # Handle merge results - use new columns if they exist, otherwise initialize
            if 'is_outlier_new' in df_cat.columns:
                df_cat['is_outlier'] = df_cat['is_outlier_new'].fillna(False)
                df_cat.drop(columns=['is_outlier_new'], inplace=True)
            elif 'is_outlier' not in df_cat.columns:
                df_cat['is_outlier'] = False
            else:
                df_cat['is_outlier'] = df_cat['is_outlier'].fillna(False)
            
            if 'outlier_type_new' in df_cat.columns:
                df_cat['outlier_type'] = df_cat['outlier_type_new']
                df_cat.drop(columns=['outlier_type_new'], inplace=True)
            elif 'outlier_type' not in df_cat.columns:
                df_cat['outlier_type'] = None
                
            df_cat.drop(columns=['_orig_idx'], inplace=True)
            
            updated_dfs.append(df_cat)
            
            out_cnt = int(df_cat['is_outlier'].sum())
            lower_cnt = int((df_cat['outlier_type'] == 'LOWER_PRICE').sum())
            upper_cnt = int((df_cat['outlier_type'] == 'UPPER_PRICE').sum())
            
            logger.info(f"  Analyzed: {len(df_clean):,}, Outliers: {out_cnt:,} "
                       f"(Lower: {lower_cnt}, Upper: {upper_cnt})")
            
            outlier_results[cat] = {
                'total_shipments': len(df_cat),
                'analyzed_shipments': len(df_clean),
                'outlier_shipments': out_cnt,
                'lower_outlier_shipments': lower_cnt,
                'upper_outlier_shipments': upper_cnt,
                'outlier_percentage': (out_cnt / len(df_cat)) * 100 if len(df_cat) else 0.0
            }
        
        # Combine all categories
        if updated_dfs:
            result_df = pd.concat(updated_dfs, ignore_index=True)
        else:
            result_df = work_df.copy()
            result_df['is_outlier'] = False
            result_df['outlier_type'] = None
        
        return result_df, outlier_results
    
    def update_outliers_in_bigquery(self, df: pd.DataFrame, batch_size: int = 1000):
        """
        Update outlier flags in BigQuery shipment_master_taxonomy table.
        
        Args:
            df: DataFrame with shipment_id, is_outlier, and outlier_type columns
            batch_size: Number of records to update per batch
        """
        if df.empty:
            logger.warning("No data to update in BigQuery")
            return
        
        # Prepare update data
        update_data = df[['shipment_id', 'is_outlier', 'outlier_type']].copy()
        update_data['is_outlier'] = update_data['is_outlier'].astype(bool)
        
        # Deduplicate by shipment_id (keep last occurrence)
        original_count = len(update_data)
        update_data = update_data.drop_duplicates(subset=['shipment_id'], keep='last')
        if len(update_data) < original_count:
            logger.warning(f"Found {original_count - len(update_data)} duplicate shipment_ids, keeping last occurrence")
        
        logger.info(f"Preparing to update {len(update_data)} records in BigQuery...")
        
        # Create temporary table
        temp_table_id = f"{self.project_id}.{self.dataset_id}._outlier_updates_temp"
        
        logger.info(f"Creating temporary table: {temp_table_id}")
        
        load_job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            schema=[
                bigquery.SchemaField('shipment_id', 'STRING', mode='REQUIRED'),
                bigquery.SchemaField('is_outlier', 'BOOL', mode='REQUIRED'),
                bigquery.SchemaField('outlier_type', 'STRING', mode='NULLABLE'),
            ],
        )
        
        # Load data to temp table
        load_job = self.client.load_table_from_dataframe(
            update_data,
            temp_table_id,
            job_config=load_job_config
        )
        load_job.result()
        
        logger.info(f"Loaded {len(update_data)} records to temporary table")
        
        # Perform MERGE to update main table
        merge_query = f"""
        MERGE `{self.shipment_taxonomy_table}` T
        USING `{temp_table_id}` S
        ON T.shipment_id = S.shipment_id
        WHEN MATCHED THEN
          UPDATE SET
            is_outlier = S.is_outlier,
            outlier_type = S.outlier_type
        """
        
        logger.info("Executing MERGE query to update main table...")
        
        # Add timeout and better error handling
        job_config = bigquery.QueryJobConfig()
        merge_job = self.client.query(merge_query, job_config=job_config)
        
        try:
            merge_job.result(timeout=120)  # 2 minute timeout
            logger.info(f"Successfully updated {len(update_data)} records in {self.shipment_taxonomy_table}")
        except Exception as e:
            logger.error(f"MERGE failed: {e}")
            logger.info("Falling back to UPDATE with subquery approach...")
            
            # Alternative: Update using a subquery with ANY_VALUE to handle duplicates
            update_query = f"""
            UPDATE `{self.shipment_taxonomy_table}` T
            SET 
              is_outlier = S.is_outlier,
              outlier_type = S.outlier_type
            FROM (
              SELECT 
                shipment_id,
                ANY_VALUE(is_outlier) as is_outlier,
                ANY_VALUE(outlier_type) as outlier_type
              FROM `{temp_table_id}`
              GROUP BY shipment_id
            ) S
            WHERE T.shipment_id = S.shipment_id
            """
            
            update_job = self.client.query(update_query)
            update_job.result(timeout=120)
            logger.info(f"Successfully updated records using UPDATE approach")
        
        # Clean up temporary table
        self.client.delete_table(temp_table_id, not_found_ok=True)
        logger.info(f"Deleted temporary table: {temp_table_id}")
    
    def analyze_product(
        self,
        product_name: str,
        category_col: str = 'auto',
        update_bq: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete outlier analysis pipeline for a single product.
        
        Args:
            product_name: Name of the product to analyze
            category_col: Column name for category-level grouping. 
                         Use 'auto' to automatically detect the best attribute.
            update_bq: Whether to update BigQuery with results
            
        Returns:
            Tuple of (analyzed DataFrame, results dictionary)
        """
        logger.info("="*80)
        logger.info(f"ANALYZING PRODUCT: {product_name}")
        logger.info("="*80)
        
        # Fetch shipment data
        df = self.get_shipments_for_product(product_name)
        
        if df.empty:
            logger.warning(f"No shipment data found for product: {product_name}")
            return df, {}
        
        # Apply filters - returns filtered data and excluded data marked as outliers
        df_filtered, df_threshold_outliers = self.apply_filters(df)
        
        if df_filtered.empty:
            logger.warning(f"No data remaining after filters for product: {product_name}")
            # Still return the threshold outliers for BQ update
            if update_bq and not df_threshold_outliers.empty:
                self.update_outliers_in_bigquery(df_threshold_outliers)
            return df_threshold_outliers, {}
        
        # Detect outliers (with auto category detection)
        df_analyzed, results = self.detect_outliers_by_category(
            df_filtered,
            category_col=category_col,
            span=self.span,
            mad_multiplier=self.mad_multiplier,
            median_pct_window=self.median_pct_window
        )
        
        # Combine analyzed data with threshold outliers
        if not df_threshold_outliers.empty:
            df_combined = pd.concat([df_analyzed, df_threshold_outliers], ignore_index=True)
        else:
            df_combined = df_analyzed
        
        # Update BigQuery with all records (analyzed + threshold outliers)
        if update_bq and not df_combined.empty:
            self.update_outliers_in_bigquery(df_combined)
        
        # Summary (only for analyzed records, not threshold filtered)
        total_outliers = int(df_analyzed['is_outlier'].sum())
        threshold_outliers = len(df_threshold_outliers)
        
        logger.info("\n" + "="*80)
        logger.info(f"PRODUCT SUMMARY: {product_name}")
        logger.info(f"Total shipments analyzed: {len(df_analyzed):,}")
        logger.info(f"Total outliers detected: {total_outliers:,} "
                   f"({total_outliers / len(df_analyzed) * 100:.2f}%)")
        if threshold_outliers > 0:
            logger.info(f"Threshold-filtered outliers: {threshold_outliers:,}")
            logger.info(f"Grand total outliers: {total_outliers + threshold_outliers:,} / {len(df_combined):,} "
                       f"({(total_outliers + threshold_outliers) / len(df_combined) * 100:.2f}%)")
        logger.info("="*80 + "\n")
        
        return df_combined, results
    
    def analyze_all_products(
        self,
        category_col: str = 'auto',
        update_bq: bool = True,
        product_limit: Optional[int] = None
    ) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
        """
        Analyze all products in product_master.
        
        Args:
            category_col: Column name for category-level grouping.
                         Use 'auto' to automatically detect the best attribute for each product.
            update_bq: Whether to update BigQuery with results
            product_limit: Maximum number of products to process (for testing)
            
        Returns:
            Dictionary mapping product_name to (DataFrame, results) tuples
        """
        logger.info("="*80)
        logger.info("STARTING OUTLIER ANALYSIS FOR ALL PRODUCTS")
        if category_col == 'auto':
            logger.info("Using AUTO category detection for each product")
        logger.info("="*80)
        
        # Get all products
        products_df = self.get_all_products()
        
        if product_limit:
            products_df = products_df.head(product_limit)
            logger.info(f"Limited to first {product_limit} products for testing")
        
        results_by_product = {}
        successful = 0
        failed = 0
        
        for idx, row in products_df.iterrows():
            product_name = row['product_name']
            
            try:
                df_analyzed, results = self.analyze_product(
                    product_name=product_name,
                    category_col=category_col,
                    update_bq=update_bq
                )
                results_by_product[product_name] = (df_analyzed, results)
                successful += 1
                
            except Exception as e:
                logger.error(f"Error analyzing product {product_name}: {str(e)}", exc_info=True)
                failed += 1
                continue
        
        logger.info("\n" + "="*80)
        logger.info("OVERALL ANALYSIS COMPLETE")
        logger.info(f"Total products: {len(products_df)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info("="*80)
        
        return results_by_product


def main():
    """
    Main entry point for outlier analysis.
    """
    # Initialize analyzer with default parameters
    analyzer = OutlierAnalyzer(
        project_id="dev-tradyon-data",
        dataset_id="tradyon",
        span=600,
        mad_multiplier=3.0,
        median_pct_window=0.30,
        value_threshold=2500.0,
        weight_threshold=500.0,
        max_unit_price=None
    )
    
    # Run analysis for all products
    results = analyzer.analyze_all_products(
        category_col='category',
        update_bq=True,
        product_limit=None  # Set to a number for testing, None for all products
    )
    
    logger.info("Outlier analysis completed successfully!")


if __name__ == "__main__":
    main()
