import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import json
import logging

logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self):
        # Configure matplotlib for non-interactive use
        import matplotlib
        matplotlib.use('Agg')
        plt.style.use('default')
    
    def create_plot_datauri(self, fig, format='png', max_size_kb=100):
        """Convert matplotlib figure to base64 data URI under size limit"""
        
        # Try different DPI values to stay under size limit
        dpi_values = [100, 80, 60, 50, 40]
        
        for dpi in dpi_values:
            try:
                # Save figure to bytes
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format=format, dpi=dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                img_buffer.seek(0)
                
                # Encode as base64
                img_bytes = img_buffer.getvalue()
                img_buffer.close()
                
                # Check size
                size_kb = len(img_bytes) / 1024
                logger.debug(f"Plot size at DPI {dpi}: {size_kb:.1f} KB")
                
                if size_kb <= max_size_kb:
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    data_uri = f"data:image/{format};base64,{img_base64}"
                    return data_uri
                
            except Exception as e:
                logger.warning(f"Failed to create plot at DPI {dpi}: {str(e)}")
                continue
        
        # If all DPI values failed, try with very low DPI
        try:
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format=format, dpi=30, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            img_buffer.seek(0)
            img_bytes = img_buffer.getvalue()
            img_buffer.close()
            
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            data_uri = f"data:image/{format};base64,{img_base64}"
            return data_uri
            
        except Exception as e:
            logger.error(f"Failed to create plot: {str(e)}")
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    def calculate_correlation(self, x, y):
        """Calculate correlation coefficient"""
        try:
            correlation = np.corrcoef(x, y)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def create_scatterplot_with_regression(self, x, y, xlabel='X', ylabel='Y', title='Scatterplot'):
        """Create scatterplot with regression line"""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot
        ax.scatter(x, y, alpha=0.6, s=50)
        
        # Regression line
        try:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", linewidth=2, alpha=0.8)
        except:
            logger.warning("Could not fit regression line")
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def analyze_csv_data(self, filepath):
        """Analyze CSV data and return basic statistics"""
        try:
            df = pd.read_csv(filepath)
            
            analysis = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing CSV: {str(e)}")
            return {'error': f'Could not analyze CSV: {str(e)}'}
