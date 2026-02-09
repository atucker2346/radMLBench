import flwr as fl
from typing import List, Tuple, Dict
import numpy as np

def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Aggregate metrics weighted by number of samples."""
    total_examples = sum([num_examples for num_examples, _ in metrics])
    if total_examples == 0:
        return {}
    
    aggregated = {}
    for num_examples, metric_dict in metrics:
        weight = num_examples / total_examples
        for key, value in metric_dict.items():
            if isinstance(value, (int, float)):
                if key not in aggregated:
                    aggregated[key] = 0.0
                aggregated[key] += value * weight
    
    return aggregated

class CrossSiloStrategy(fl.server.strategy.FedAvg):
    """Custom strategy for cross-silo federated learning."""
    
    def aggregate_fit(self, rnd: int, results: List[Tuple], failures: List):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_metrics:
            print(f"\nRound {rnd} - Training:")
            print(f"  Sites: {len(results)}")
            for client_id, fit_res in results:
                # Handle both tuple and object formats
                if isinstance(fit_res, tuple):
                    # If it's a tuple, unpack it
                    num_examples, metrics = fit_res
                else:
                    # If it's an object, access attributes
                    num_examples = fit_res.num_examples
                    metrics = fit_res.metrics if fit_res.metrics else {}
                
                site_name = metrics.get('site_name', f'Client_{client_id}')
                train_auc = metrics.get('train_auc', 0.0)
                print(f"    Site {client_id} ({site_name}): AUC={train_auc:.4f}, Samples={num_examples}")
            print(f"  Aggregated AUC: {aggregated_metrics.get('train_auc', 0.0):.4f}")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, rnd: int, results: List[Tuple], failures: List):
        if not results:
            return None, {}
        
        # Handle tuple format: (client_id, EvaluateRes) or (client_id, loss, num_examples, metrics)
        losses = []
        num_examples_list = []
        metrics_list = []
        
        for client_id, result in results:
            # Check if result is an EvaluateRes object
            if hasattr(result, 'loss'):
                # It's an EvaluateRes object
                losses.append(result.loss if result.loss is not None else None)
                num_examples_list.append(result.num_examples)
                metrics_dict = result.metrics if result.metrics else {}
                metrics_list.append((result.num_examples, metrics_dict))
            elif isinstance(result, tuple):
                # It's a tuple: (loss, num_examples, metrics)
                loss, num_examples, metrics_dict = result
                losses.append(loss if loss is not None else None)
                num_examples_list.append(num_examples)
                metrics_dict = metrics_dict if metrics_dict else {}
                metrics_list.append((num_examples, metrics_dict))
            else:
                # Fallback: assume it's (loss, num_examples, metrics)
                try:
                    loss, num_examples, metrics_dict = result
                    losses.append(loss if loss is not None else None)
                    num_examples_list.append(num_examples)
                    metrics_dict = metrics_dict if metrics_dict else {}
                    metrics_list.append((num_examples, metrics_dict))
                except:
                    # If unpacking fails, skip this result
                    continue
        
        # Calculate aggregated loss
        valid_losses = [l for l in losses if l is not None]
        if valid_losses:
            aggregated_loss = np.average(
                valid_losses,
                weights=[num_examples_list[i] for i, l in enumerate(losses) if l is not None]
            )
        else:
            aggregated_loss = None
        
        # Aggregate metrics
        aggregated_metrics = weighted_average(metrics_list)
        
        print(f"\nRound {rnd} - Evaluation:")
        print(f"  Sites: {len(results)}")
        for client_id, result in enumerate(results):
            # Extract metrics for printing
            if hasattr(result, 'metrics'):
                metrics_dict = result.metrics if result.metrics else {}
                num_examples = result.num_examples
            elif isinstance(result, tuple) and len(result) >= 3:
                _, num_examples, metrics_dict = result[:3]
                metrics_dict = metrics_dict if metrics_dict else {}
            else:
                metrics_dict = {}
                num_examples = 0
            
            site_name = metrics_dict.get('site_name', f'Client_{client_id}')
            test_auc = metrics_dict.get('test_auc', 0.0)
            print(f"    Site {client_id} ({site_name}): AUC={test_auc:.4f}, Samples={num_examples}")
        print(f"  Aggregated AUC: {aggregated_metrics.get('test_auc', 0.0):.4f}")
        
        return aggregated_loss, aggregated_metrics

strategy = CrossSiloStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)
