try:
    import pyproj
    from pyproj import Transformer
    print(f"pyproj imported successfully: {pyproj.__version__}")
    
    import rasterio
    print(f"rasterio imported successfully: {rasterio.__version__}")
    
    import tables
    print(f"tables imported successfully: {tables.__version__}")
    
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
