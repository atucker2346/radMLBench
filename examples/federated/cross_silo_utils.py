import numpy as np
import pandas as pd
import radMLBench
from typing import List, Dict
from collections import defaultdict

class SiteBasedSplitter:
    """Split data by site/institution for cross-silo federated learning."""
    
    def __init__(self, split_method: str = "by_dataset"):
        self.split_method = split_method
    
    def split_by_dataset(self, dataset_names: List[str], local_cache_dir: str = "./datasets") -> Dict[int, Dict]:
        """Each dataset represents a different site/hospital."""
        sites = {}
        
        for site_id, dataset_name in enumerate(dataset_names):
            X, y = radMLBench.loadData(
                dataset_name, 
                return_X_y=True, 
                local_cache_dir=local_cache_dir
            )
            
            meta = radMLBench.getMetaData(dataset_name)
            
            sites[site_id] = {
                'X': X,
                'y': y,
                'site_name': dataset_name,
                'metadata': meta,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'class_balance': np.mean(y),
                'modality': meta.get('modality', 'Unknown'),
                'pathology': meta.get('pathology', 'Unknown')
            }
        
        return sites
    
    def align_features(self, sites: Dict[int, Dict], alignment_method: str = "intersection") -> Dict[int, Dict]:
        """Align features across sites."""
        feature_counts = [site['n_features'] for site in sites.values()]
        
        if alignment_method == "intersection":
            min_features = min(feature_counts)
            for site_id in sites:
                sites[site_id]['X'] = sites[site_id]['X'][:, :min_features]
                sites[site_id]['n_features'] = min_features
        
        return sites
    
    def get_site_statistics(self, sites: Dict[int, Dict]) -> pd.DataFrame:
        """Get statistics about each site."""
        stats = []
        for site_id, site_data in sites.items():
            stats.append({
                'site_id': site_id,
                'site_name': site_data['site_name'],
                'n_samples': site_data['n_samples'],
                'n_features': site_data['n_features'],
                'class_balance': site_data['class_balance'],
                'modality': site_data['modality'],
                'pathology': site_data['pathology']
            })
        return pd.DataFrame(stats)
