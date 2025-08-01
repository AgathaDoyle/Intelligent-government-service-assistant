from model_loader import get_components
import joblib

components = get_components()
logger = components['logger']
paths = components['paths']

# history_features_path = paths['history_features_path']
# history_labels_path = paths['history_labels_path']
#
# history_features = joblib.load(history_features_path)
# history_labels = joblib.load(history_labels_path)

for k, v in paths.items():
    print(k, ":\t", v)
