import SimpleITK as sitk
import os
import glob
from config import STANFORD_DATA_DIR

def convert_stanford_pngs_to_nifti(stanford_dir_path: str = STANFORD_DATA_DIR):
    base_split_dir = os.path.join(stanford_dir_path, 'stanford_release_brainmask', 'mets_stanford_releaseMask_train')
    all_dirs = glob.glob(base_split_dir + '/*')

    for met_dir in all_dirs:
        for mod in ['0', '1', '2', '3', 'seg']:
            mod_dir = os.path.join(met_dir, mod)
            if os.path.exists(mod_dir):
                images_list = os.listdir(mod_dir)
                images_sorted = sorted(images_list, key=lambda x: int(os.path.splitext(x)[0]))
                image_dirs_sorted = [os.path.join(mod_dir, image_file) for image_file in images_sorted]

                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(image_dirs_sorted)
                vol = reader.Execute()
                vol.SetSpacing((0.94, 0.94, 1.0))
                
                nifti_path = os.path.join(mod_dir, 'volume.nii.gz')
                print(nifti_path)
                sitk.WriteImage(vol, nifti_path)

if __name__ == "__main__":
    convert_stanford_pngs_to_nifti(STANFORD_DATA_DIR)