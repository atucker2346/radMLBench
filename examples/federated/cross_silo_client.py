import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Dict

class CrossSiloClient(fl.client.NumPyClient):
    """Client for cross-silo federated learning."""
    
    def __init__(self, site_id: int, site_data: Dict):
        self.site_id = site_id
        self.site_name = site_data['site_name']
        
        # Split data (80/20)
        X = site_data['X']
        y = site_data['y']
        split_idx = int(0.8 * len(X))
        
        self.X_train = X[:split_idx]
        self.y_train = y[:split_idx]
        self.X_test = X[split_idx:]
        self.y_test = y[split_idx:]
        
        # Normalize
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize neural network."""
        input_dim = self.X_train.shape[1]
        
        class SiteNN(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.dropout1 = nn.Dropout(0.3)
                self.fc2 = nn.Linear(128, 64)
                self.dropout2 = nn.Dropout(0.3)
                self.fc3 = nn.Linear(64, 32)
                self.fc4 = nn.Linear(32, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout1(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout2(x)
                x = torch.relu(self.fc3(x))
                x = self.sigmoid(self.fc4(x))
                return x
        
        self.model = SiteNN(input_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=False)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=config.get('lr', 0.001))
        criterion = nn.BCELoss()
        
        X_tensor = torch.FloatTensor(self.X_train).to(self.device)
        y_tensor = torch.FloatTensor(self.y_train).to(self.device)
        
        epochs = config.get('epochs', 5)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor).squeeze()
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            train_pred = self.model(X_tensor).squeeze().cpu().numpy()
            train_auc = roc_auc_score(self.y_train, train_pred)
        
        num_examples = len(self.X_train)
        metrics = {
            'train_auc': float(train_auc),
            'site_id': self.site_id,
            'site_name': self.site_name
        }
        
        return self.get_parameters(config), num_examples, metrics
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
        
        with torch.no_grad():
            test_pred = self.model(X_test_tensor).squeeze().cpu().numpy()
            test_auc = roc_auc_score(self.y_test, test_pred)
            test_acc = accuracy_score(self.y_test, (test_pred > 0.5).astype(int))
        
        num_examples = len(self.X_test)
        metrics = {
            'test_auc': float(test_auc),
            'test_accuracy': float(test_acc),
            'site_id': self.site_id,
            'site_name': self.site_name
        }
        
        return float(test_auc), num_examples, metrics
