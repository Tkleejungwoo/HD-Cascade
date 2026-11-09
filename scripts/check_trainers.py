"""Check imports and schedule boundaries without training or inference."""

from pathlib import Path
import nnunetv2
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerHDNoMirroring import nnUNetTrainerHDNoMirroring
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerHDWithMirroring import nnUNetTrainerHDWithMirroring
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBasePlusHDCube import nnUNetTrainerBasePlusHDCube, hd_weight_for_epoch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerHDCascade2000 import nnUNetTrainerHDCascade2000
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


def main():
    trainer_folder = Path(nnunetv2.__file__).resolve().parent / 'training' / 'nnUNetTrainer'
    for cls in (nnUNetTrainerHDNoMirroring, nnUNetTrainerHDWithMirroring,
                nnUNetTrainerBasePlusHDCube, nnUNetTrainerHDCascade2000):
        assert issubclass(cls, nnUNetTrainer)
        found = recursive_find_python_class(str(trainer_folder), cls.__name__,
                                           'nnunetv2.training.nnUNetTrainer')
        assert found is cls, f'nnU-Net cannot discover {cls.__name__}'
        print(f'Import and trainer discovery OK: {cls.__name__}')
    assert hd_weight_for_epoch(1001) == 0.0
    assert 0 < hd_weight_for_epoch(1999) < 0.1
    assert hd_weight_for_epoch(2000) == 0.1
    print('HD-weight schedule boundaries OK')


if __name__ == '__main__':
    main()
