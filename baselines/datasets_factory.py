from baselines.datasets import *
from baselines.supported_models import TRAINING_SUPPORTED_MODELS_5_TRACKS, TRAINING_SUPPORTED_MODELS_6_TRACKS, PROGRAMMED_MODELS, WORSER_MODELS


class DatasetsFactory(object):

    @staticmethod
    def get_training_dataset(model_name: str, samples_dir: str, labels_dir: str, mask_file_path: str, num_frames: int, prefixes: list = []) -> CaterAbstractDataset:

        if model_name in TRAINING_SUPPORTED_MODELS_5_TRACKS:
            return Cater5TracksForObjectsTrainingDataset(samples_dir, labels_dir, mask_file_path, num_frames, prefixes)

        elif model_name in TRAINING_SUPPORTED_MODELS_6_TRACKS:
            return Cater6TracksForObjectsTrainingDataset(samples_dir, labels_dir, mask_file_path, num_frames, prefixes)

        elif model_name in WORSER_MODELS:
            return WorserDataset(samples_dir, labels_dir, mask_file_path, num_frames)

    @staticmethod
    def get_inference_dataset(model_name: str, samples_dir: str, labels_dir: str, prefixes: list) -> CaterAbstractDataset:

        if model_name in PROGRAMMED_MODELS + TRAINING_SUPPORTED_MODELS_5_TRACKS:
            return Cater5TracksForObjectsInferenceDataset(samples_dir, labels_dir)

        if model_name in TRAINING_SUPPORTED_MODELS_6_TRACKS:
			#AT: It uses this one!
            return Cater6TracksForObjectsInferenceDataset(samples_dir, labels_dir, prefixes)

        if model_name in WORSER_MODELS:
            return WorserInferenceDataset(samples_dir, labels_dir)
