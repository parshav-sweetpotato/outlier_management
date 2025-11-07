#!/usr/bin/env python3
"""
Command-line script to run outlier analysis.

Usage:
    python run_analysis.py --all                    # Analyze all products
    python run_analysis.py --product "cloves"       # Analyze specific product
    python run_analysis.py --all --test --limit 5   # Test mode with 5 products
"""

import argparse
import sys
from outlier_analyzer import OutlierAnalyzer
from config import (
    BIGQUERY_CONFIG,
    OUTLIER_PARAMS,
    PROCESSING_CONFIG,
    CATEGORY_COLUMN,
    get_config_for_product
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run outlier detection analysis on shipment data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all products and update BigQuery
  python run_analysis.py --all

  # Analyze a specific product
  python run_analysis.py --product "cloves"

  # Test mode: analyze first 5 products without updating BigQuery
  python run_analysis.py --all --test --limit 5

  # Analyze with custom parameters
  python run_analysis.py --all --span 800 --mad-multiplier 2.5
        """
    )
    
    # Product selection
    product_group = parser.add_mutually_exclusive_group(required=True)
    product_group.add_argument(
        '--all',
        action='store_true',
        help='Analyze all products in product_master'
    )
    product_group.add_argument(
        '--product',
        type=str,
        help='Analyze a specific product by name'
    )
    
    # Testing options
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: do not update BigQuery'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of products to process (for testing)'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--span',
        type=int,
        default=OUTLIER_PARAMS.get('span', 600),
        help=f'EWM span for smoothness (default: {OUTLIER_PARAMS.get("span", 600)})'
    )
    parser.add_argument(
        '--mad-multiplier',
        type=float,
        default=OUTLIER_PARAMS.get('mad_multiplier', 3.0),
        help=f'MAD multiplier for bounds (default: {OUTLIER_PARAMS.get("mad_multiplier", 3.0)})'
    )
    parser.add_argument(
        '--median-pct-window',
        type=float,
        default=OUTLIER_PARAMS.get('median_pct_window', 0.30),
        help=f'Median percentage window (default: {OUTLIER_PARAMS.get("median_pct_window", 0.30)})'
    )
    
    # Filter parameters
    parser.add_argument(
        '--value-threshold',
        type=float,
        default=OUTLIER_PARAMS['value_threshold'],
        help=f'Minimum value threshold in USD (default: {OUTLIER_PARAMS["value_threshold"]})'
    )
    parser.add_argument(
        '--weight-threshold',
        type=float,
        default=OUTLIER_PARAMS['weight_threshold'],
        help=f'Minimum weight threshold in kg (default: {OUTLIER_PARAMS["weight_threshold"]})'
    )
    parser.add_argument(
        '--max-price',
        type=float,
        default=OUTLIER_PARAMS['max_unit_price'],
        help='Maximum unit price threshold (optional)'
    )
    
    # Category column
    parser.add_argument(
        '--category-col',
        type=str,
        default=CATEGORY_COLUMN,
        help=f'Category column name for grouping. Use "auto" to auto-detect best attribute (default: {CATEGORY_COLUMN})'
    )
    
    # BigQuery configuration
    parser.add_argument(
        '--project',
        type=str,
        default=BIGQUERY_CONFIG['project_id'],
        help=f'BigQuery project ID (default: {BIGQUERY_CONFIG["project_id"]})'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=BIGQUERY_CONFIG['dataset_id'],
        help=f'BigQuery dataset ID (default: {BIGQUERY_CONFIG["dataset_id"]})'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine whether to update BigQuery
    update_bq = not args.test
    
    print("="*80)
    print("OUTLIER ANALYSIS")
    print("="*80)
    print(f"Mode: {'TEST (no BigQuery updates)' if args.test else 'PRODUCTION (will update BigQuery)'}")
    
    if args.all:
        print(f"Scope: All products")
        if args.limit:
            print(f"Limit: First {args.limit} products")
    else:
        print(f"Scope: Single product - '{args.product}'")
    
    print(f"Parameters:")
    print(f"  - EWM span: {args.span}")
    print(f"  - MAD multiplier: {args.mad_multiplier}")
    print(f"  - Median % window: ±{args.median_pct_window:.1%}")
    print(f"  - Value threshold: ${args.value_threshold:,.2f}")
    print(f"  - Weight threshold: {args.weight_threshold:,.2f} kg")
    if args.max_price:
        print(f"  - Max unit price: ${args.max_price:,.2f}")
    print(f"  - Category column: {args.category_col}")
    print(f"BigQuery: {args.project}.{args.dataset}")
    print("="*80)
    print()
    
    # Confirmation prompt for production mode
    if not args.test:
        response = input("This will update BigQuery. Continue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Aborted.")
            sys.exit(0)
    
    # Initialize analyzer
    analyzer = OutlierAnalyzer(
        project_id=args.project,
        dataset_id=args.dataset,
        span=args.span,
        mad_multiplier=args.mad_multiplier,
        median_pct_window=args.median_pct_window,
        value_threshold=args.value_threshold,
        weight_threshold=args.weight_threshold,
        max_unit_price=args.max_price
    )
    
    try:
        if args.all:
            # Analyze all products
            results = analyzer.analyze_all_products(
                category_col=args.category_col,
                update_bq=update_bq,
                product_limit=args.limit
            )
            
            # Summary
            print("\n" + "="*80)
            print("SUMMARY")
            print("="*80)
            total_products = len(results)
            total_shipments = sum(len(df) for df, _ in results.values())
            total_outliers = sum(df['is_outlier'].sum() for df, _ in results.values())
            
            print(f"Products analyzed: {total_products}")
            print(f"Total shipments: {total_shipments:,}")
            print(f"Total outliers: {total_outliers:,} ({total_outliers/total_shipments*100:.2f}%)")
            print("="*80)
            
        else:
            # Analyze single product
            # Check if product-specific config exists
            product_config = get_config_for_product(args.product)
            
            # Override analyzer parameters if product-specific config exists
            if product_config != OUTLIER_PARAMS:
                print(f"\nUsing product-specific configuration for '{args.product}'")
                analyzer = OutlierAnalyzer(
                    project_id=args.project,
                    dataset_id=args.dataset,
                    span=product_config.get('span', args.span),
                    mad_multiplier=product_config.get('mad_multiplier', args.mad_multiplier),
                    median_pct_window=product_config.get('median_pct_window', args.median_pct_window),
                    value_threshold=product_config.get('value_threshold', args.value_threshold),
                    weight_threshold=product_config.get('weight_threshold', args.weight_threshold),
                    max_unit_price=product_config.get('max_unit_price', args.max_price)
                )
            
            df_analyzed, results = analyzer.analyze_product(
                product_name=args.product,
                category_col=args.category_col,
                update_bq=update_bq
            )
            
            # Display category-level results
            if results:
                print("\n" + "="*80)
                print(f"CATEGORY BREAKDOWN FOR '{args.product}'")
                print("="*80)
                for category, stats in results.items():
                    print(f"\n{category}:")
                    print(f"  Total shipments: {stats['total_shipments']:,}")
                    print(f"  Analyzed: {stats['analyzed_shipments']:,}")
                    print(f"  Outliers: {stats['outlier_shipments']:,} ({stats['outlier_percentage']:.2f}%)")
                    print(f"    - Lower: {stats['lower_outlier_shipments']:,}")
                    print(f"    - Upper: {stats['upper_outlier_shipments']:,}")
                print("="*80)
        
        print("\n✓ Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
