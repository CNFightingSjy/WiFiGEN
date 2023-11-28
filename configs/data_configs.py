from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'wifi_encode': {
		'transforms': transforms_config.MatTransforms,
		'train_source_root': dataset_paths['ptot'],
		'train_target_root': dataset_paths['epsono'],
		'test_source_root': dataset_paths['ptot_test'],
		'test_target_root': dataset_paths['epsono_test'],
	}
}
