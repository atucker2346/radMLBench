import flwr as fl
import sys
from cross_silo_utils import SiteBasedSplitter
from cross_silo_client import CrossSiloClient
from cross_silo_server import strategy
import radMLBench

def start_client(site_id: int, site_data: dict, server_address: str = "localhost:8080"):
    """Start a Flower client for a specific site."""
    client = CrossSiloClient(site_id, site_data)
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )

def start_server(num_rounds: int = 10, server_address: str = "localhost:8080"):
    """Start Flower server."""
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )

if __name__ == "__main__":
    SERVER_ADDRESS = "localhost:8080"
    NUM_ROUNDS = 10
    
    # Get brain-related datasets as different sites
    all_datasets = radMLBench.listDatasets()
    brain_datasets = []
    for dataset in all_datasets:
        meta = radMLBench.getMetaData(dataset)
        if 'brain' in meta['pathology'].lower() or \
           'glioma' in meta['pathology'].lower() or \
           'glioblastoma' in meta['pathology'].lower():
            brain_datasets.append(dataset)
    
    # Use first 2 brain datasets as 2 different sites (start small)
    site_datasets = brain_datasets[:2]
    print(f"Using {len(site_datasets)} datasets as sites: {site_datasets}")
    
    # Create site splits
    # Use absolute path to datasets directory (relative to project root)
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datasets_path = os.path.join(project_root, "datasets")
    
    splitter = SiteBasedSplitter(split_method="by_dataset")
    sites = splitter.split_by_dataset(site_datasets, local_cache_dir=datasets_path)
    sites = splitter.align_features(sites, alignment_method="intersection")
    
    # Print site statistics
    stats_df = splitter.get_site_statistics(sites)
    print("\nSite Statistics:")
    print(stats_df.to_string(index=False))
    
    # Start server or client
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        print(f"\nStarting server on {SERVER_ADDRESS}...")
        print(f"Number of rounds: {NUM_ROUNDS}")
        start_server(NUM_ROUNDS, SERVER_ADDRESS)
    else:
        # Start client
        site_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        if site_id not in sites:
            print(f"Error: Site {site_id} not found. Available sites: {list(sites.keys())}")
            sys.exit(1)
        
        site_data = sites[site_id]
        print(f"\nStarting client for Site {site_id}: {site_data['site_name']}")
        print(f"  Samples: {site_data['n_samples']}")
        print(f"  Features: {site_data['n_features']}")
        start_client(site_id, site_data, SERVER_ADDRESS)
