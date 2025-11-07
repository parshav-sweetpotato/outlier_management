"""Configuration for outlier detection system."""

# BigQuery Configuration
BIGQUERY_CONFIG = {
    'project_id': 'dev-tradyon-data',
    'dataset_id': 'tradyon',
    'product_master_table': 'product_master',
    'shipment_taxonomy_table': 'shipment_master_taxonomy',
}

# Outlier Detection Parameters
OUTLIER_PARAMS = {
    'span': 600,                 # EWM smoothness
    'mad_multiplier': 3.5,       # MAD bounds width
    'median_pct_window': 0.40,   # ±40% median window
    'value_threshold': 1000.0,   # Min value USD
    'weight_threshold': 500.0,   # Min weight kg
    'max_unit_price': None,      # Max price (optional)
}

# Category column: 'auto' for auto-detection, or specify attribute name
CATEGORY_COLUMN = 'auto'

# Processing Configuration
PROCESSING_CONFIG = {
    'batch_size': 1000,
    'product_limit': None,
    'update_bigquery': True,
}

# Product-specific overrides
PRODUCT_SPECIFIC_CONFIG = {
    'cloves': {
        'max_unit_price': 100.0,
        'weight_threshold': 1000.0,
    },
    'coffee': {
        'max_unit_price': 100.0,
        'weight_threshold': 1000.0,
    },
    'Black Pepper': {
        'max_unit_price': 100.0,
        'weight_threshold': 1000.0,
    },
}


def get_config_for_product(product_name: str) -> dict:
    """Get configuration for specific product."""
    config = OUTLIER_PARAMS.copy()
    product_key = product_name.lower()
    if product_key in PRODUCT_SPECIFIC_CONFIG:
        config.update(PRODUCT_SPECIFIC_CONFIG[product_key])
    return config
