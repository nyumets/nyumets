from torch.utils.data import get_worker_info

import monai
from monai.data import Dataset, IterableDataset
from monai.transforms import apply_transform
from monai.transforms.transform import Randomizable
from typing import Dict, Optional, Tuple, Type, Union, Sequence, IO, TYPE_CHECKING, Any, Callable, List
import random

class TemporalIterableDataset(IterableDataset):
    """
    """
    def __init__(
        self,
        data: Sequence, 
        transform: Optional[Callable] = None,
        partition_by: str = 'patient_id',
        sort_by: str = 'timepoint',
        store_previous: bool = False,  # use for validation
        combine_timepoints: bool = False,  # use for STT training
        sequence_limit: int = 5,
        registration_mri_sequence: str = 'T1_post'
    ):
        
        self.transform = transform
        self.partition_by = partition_by
        self.sort_by = sort_by
        self.combine_timepoints = combine_timepoints
        self.store_previous = store_previous
        self.sequence_limit = sequence_limit
        self.registration_mri_sequence = registration_mri_sequence
        
        self.data = self._partition_temporal_data(data)
        self.data_len = self._calculate_data_len()
        
        super().__init__(self.data, self.transform)
    
    def __len__(self):
        return self.data_len
    
    def __iter__(self):
        info = get_worker_info()
        num_workers = info.num_workers if info is not None else 1
        id = info.id if info is not None else 0

        self.source = iter(self.data)
        for i, item in enumerate(self.source):
            if self.combine_timepoints:
                # will output dynamic batches with all image/label sequences for a given patient
                if i % num_workers == id:
                    if self.transform is not None:
                        data_dict = apply_transform(self.transform, item)
                    yield data_dict
            else:
                for j, data_dict in enumerate(item):
                    if j % num_workers == id:
                        if self.transform is not None:
                            data_dict = apply_transform(self.transform, data_dict)
                        yield data_dict
    
    def _calculate_data_len(self):
        if self.combine_timepoints:
             return len(self.data)
        else:
            total_len = 0
            self.source = iter(self.data)
            for i, item in enumerate(self.source):
                total_len += len(item)
            return total_len


    def _partition_temporal_data(
        self,
        data: dict
    ):
        """
        Partition data by patient and sequential position.
        
        """
        # Group the data entries
        partitioned_dict = {}
        for data_dict in data:
            p = data_dict[self.partition_by]
            if p in partitioned_dict.keys():
                partitioned_dict[p].append(data_dict)
            else:
                partitioned_dict[p] = [data_dict]

        # Sort data by sequential position in time
        partitioned_data_list = []
        if self.sort_by is not None:
            partitioned_sorted_dict = {}
            
            for p_id, p_list in partitioned_dict.items():
                newlist = sorted(p_list, key=lambda d: d[self.sort_by])
                
                if self.store_previous:
                    # TODO: find a less memory-intensive way 
                    if self.registration_mri_sequence in newlist[0].keys():
                        previous_image_path = newlist[0][self.registration_mri_sequence]
                        previous_label_path = newlist[0]['label']

                        for relative_index, timepoint_dict in enumerate(newlist):
                            timepoint_dict['relative_timepoint'] = relative_index
                            timepoint_dict['prev_image'] = previous_image_path
                            timepoint_dict['prev_label'] = previous_label_path
                            previous_image_path = timepoint_dict[self.registration_mri_sequence]
                            previous_label_path = timepoint_dict['label']
            
                if len(newlist) > self.sequence_limit:
                    newlist = newlist[:self.sequence_limit]

                partitioned_data_list.append(newlist)
        
        else:
            for p_id, p_list in partitioned_dict.items():
                partitioned_data_list.append(p_list)

        return partitioned_data_list


class TemporalShuffleBuffer(Randomizable, TemporalIterableDataset):
    """
    Extend the IterableDataset with a buffer and randomly pop items.

    Args:
        data: input data source to load and transform to generate dataset for model.
        transform: a callable data transform on input data.
        buffer_size: size of the buffer to store items and randomly pop, default to 512.
        seed: random seed to initialize the random state of all workers, set `seed += 1` in
            every iter() call, refer to the PyTorch idea:
            https://github.com/pytorch/pytorch/blob/v1.10.0/torch/utils/data/distributed.py#L98.

    """

    def __init__(self, data, transform=None, buffer_size: int = 512, seed: int = 0, **kwargs) -> None:
        super().__init__(data=data, transform=transform, **kwargs)
        self.size = buffer_size
        self.seed = seed
        self._idx = 0

    def __iter__(self):
        """
        Fetch data from the source, if buffer is not full, fill into buffer, otherwise,
        randomly pop items from the buffer.
        After loading all the data from source, randomly pop items from the buffer.

        """
        self.seed += 1
        super().set_random_state(seed=self.seed)  # make all workers in sync
        buffer = []
        source = self.data

        def _pop_item():
            self.randomize(len(buffer))
            # switch random index data and the last index data
            ret, buffer[self._idx] = buffer[self._idx], buffer[-1]
            buffer.pop()
            return ret

        def _get_item():
            for item in source:
                if len(buffer) >= self.size:
                    yield _pop_item()
                buffer.append(item)

            while buffer:
                yield _pop_item()

        self.data = _get_item()
        return super().__iter__()

    def randomize(self, size: int) -> None:
        self._idx = self.R.randint(size)


TemporalDataset = TemporalIterableDataset
