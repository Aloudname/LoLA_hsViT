import os
import re
import sys
import glob
import logging
import numpy as np
import scipy.io as sio

from tqdm import tqdm
from scipy.io import savemat
from typing import Dict, Tuple, Optional, List


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('envi2mat_batch.log'), logging.StreamHandler()]
)


class ENVI2MATConverter:
    """
    ENVI -> .mat converter.
    """
    ENVI_DTYPE_MAP = {
        1: np.uint8,    # data type=1 -> uint8
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        6: np.complex64,
        9: np.complex128,
        12: np.uint16,
        13: np.uint32,
        14: np.int64,
        15: np.uint64
    }

    def __init__(self, envi_hdr_path: str, envi_data_path: Optional[str] = None):
        self.hdr_path = envi_hdr_path
        self.data_path = envi_data_path if envi_data_path else envi_hdr_path.replace('.hdr', '')
        self.hdr_info = self._parse_hdr()

    def _parse_hdr(self) -> Dict:
        """
            .hdr unpacker.
        """
        hdr_info = {}
        try:
            with open(self.hdr_path, 'r', encoding='utf-8') as f:
                hdr_content = f.read().lower()

            # core keys.
            hdr_info['samples'] = int(self._extract_hdr_field(hdr_content, r'samples\s*=\s*(\d+)'))
            hdr_info['lines'] = int(self._extract_hdr_field(hdr_content, r'lines\s*=\s*(\d+)'))
            hdr_info['bands'] = int(self._extract_hdr_field(hdr_content, r'bands\s*=\s*(\d+)'))
            hdr_info['data_type'] = int(self._extract_hdr_field(hdr_content, r'data type\s*=\s*(\d+)'))
            hdr_info['interleave'] = self._extract_hdr_field(hdr_content, r'interleave\s*=\s*(\w+)')
            hdr_info['byte_order'] = int(self._extract_hdr_field(hdr_content, r'byte order\s*=\s*(\d+)', default='0'))

            # wavelength.
            wavelength_str = self._extract_hdr_field(hdr_content, r'wavelength\s*=\s*\{(.*?)\}', default='')
            hdr_info['wavelength'] = np.array([float(x.strip()) for x in wavelength_str.split(',') if x.strip()]) if wavelength_str else None

            # confirm data type.
            if hdr_info['data_type'] not in self.ENVI_DTYPE_MAP:
                raise ValueError(f"Invalid data type: {hdr_info['data_type']}")
            hdr_info['numpy_dtype'] = self.ENVI_DTYPE_MAP[hdr_info['data_type']]

            return hdr_info
        except Exception as e:
            logging.error(f"Fail loading {self.hdr_path}: {str(e)}")
            raise

    def _extract_hdr_field(self, hdr_content: str, pattern: str, default: Optional[str] = None) -> str:
        match = re.findall(pattern, hdr_content)
        if match:
            return match[0]
        if default is not None:
            return default
        raise ValueError(f"No match in .hdr for: {pattern}")

    def _read_envi_data(self) -> np.ndarray:
        try:
            lines, samples, bands = self.hdr_info['lines'], self.hdr_info['samples'], self.hdr_info['bands']
            dtype = self.hdr_info['numpy_dtype']
            interleave = self.hdr_info['interleave']
            byte_order = self.hdr_info['byte_order']

            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Cannot find {self.data_path}")
            data = np.fromfile(self.data_path, dtype=dtype)
            logging.info(f"Reading {self.data_path} | primary length: {len(data)} | dtype: {dtype}")

            # byteorder adjust (int/float)
            if byte_order != (sys.byteorder == 'little'):
                data = data.byteswap()
                logging.info(f"Byteorder adjusted (byte_order={byte_order})")

            # reconstruct data dim.
            expected_length = lines * samples * bands
            if len(data) != expected_length:
                raise ValueError(
                    f"Data length mismatch! Expecting: {expected_length}({lines}x{samples}x{bands}), actual: {len(data)}"
                )

            if interleave == 'bsq':
                data = data.reshape((bands, lines, samples)).transpose(1, 2, 0)
            elif interleave == 'bil':
                data = data.reshape((lines, bands, samples)).transpose(0, 2, 1)
            elif interleave == 'bip':
                data = data.reshape((lines, samples, bands))
            else:
                raise ValueError(f"Unsupported type: {interleave}")

            # confirm data type (uint8 for 0-255)
            if self.hdr_info['data_type'] == 1:
                if np.min(data) < 0 or np.max(data) > 255:
                    logging.warning(f"Pixel value incorrect! min={np.min(data)}, max={np.max(data)}")
                    # adjust to 0-255
                    data = np.clip(data, 0, 255)
                    logging.info(f"Cut pixel value to 0-255 (uint8)")

            # check NaN/Inf
            if np.issubdtype(dtype, np.floating):
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    logging.warning("Data contains NaN/Inf, replaced with 0.")
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            return data
        except Exception as e:
            logging.error(f"Failed loading data {self.data_path}: {str(e)}")
            raise

    def convert_to_mat(self, mat_save_path: str, save_hdr_info: bool = True) -> None:
        """Convert .hdr to .mat"""
        try:
            hyperspectral_data = self._read_envi_data()

            mat_dict = {'hyperspectral_data': hyperspectral_data}
            if save_hdr_info:
                mat_dict['hdr_info'] = {
                    'samples': self.hdr_info['samples'],
                    'lines': self.hdr_info['lines'],
                    'bands': self.hdr_info['bands'],
                    'data_type': self.hdr_info['data_type'],
                    'interleave': self.hdr_info['interleave'],
                    'byte_order': self.hdr_info['byte_order'],
                    'wavelength': self.hdr_info['wavelength'] if self.hdr_info['wavelength'] is not None else np.array([])
                }

            os.makedirs(os.path.dirname(mat_save_path), exist_ok=True)
            savemat(mat_save_path, mat_dict)

            logging.info(f"Convert success:{self.hdr_path} -> {mat_save_path}")
            logging.info(f"Data dim: {hyperspectral_data.shape} | dtype: {hyperspectral_data.dtype}")
            logging.info(f"Data range: min={np.min(hyperspectral_data)}, max={np.max(hyperspectral_data)}")

        except Exception as e:
            logging.error(f"Failed converting {self.hdr_path}: {str(e)}")
            raise

    @classmethod
    def batch_convert(cls, input_dir: str, output_dir: str, hdr_suffix: str = '.hdr',
                      data_suffixes: List[str] = None, save_hdr_info: bool = True) -> None:
        hdr_files = glob.glob(os.path.join(input_dir, f'*{hdr_suffix}'))
        if not hdr_files:
            logging.warning(f"Cannot find {hdr_suffix}: {input_dir}")
            return

        success_count = 0
        fail_count = 0
        with tqdm(total=len(hdr_files), desc="ENVI -> MAT batch convert") as pbar:
            for hdr_file in hdr_files:
                try:
                    data_file = None
                    if data_suffixes:
                        for suffix in data_suffixes:
                            candidate = hdr_file.replace(hdr_suffix, suffix)
                            if os.path.exists(candidate):
                                data_file = candidate
                                break
                    if data_file is None:
                        data_file = hdr_file.replace(hdr_suffix, '')

                    converter = cls(envi_hdr_path=hdr_file, envi_data_path=data_file)
                    file_name = os.path.basename(hdr_file).replace(hdr_suffix, '.mat')
                    mat_save_path = os.path.join(output_dir, file_name)
                    converter.convert_to_mat(mat_save_path, save_hdr_info)
                    success_count += 1

                except Exception as e:
                    fail_count += 1
                    logging.error(f"Failed converting {hdr_file}: {str(e)}")
                finally:
                    pbar.update(1)

        logging.info(f"\nSuccessful convertion! | Total num: {len(hdr_files)} | succeeded: {success_count} | 失败：{fail_count}")
        logging.info(f"Output dir: {os.path.abspath(output_dir)}")

def read_check(sample_name):
    mat_data = sio.loadmat(f"/data/chenhaoran/WHUhypspec/data/WHU-Hi-{sample_name}_gt.mat")
    hs_data = mat_data['hyperspectral_data']

    print(f"\nComfirming {sample_name}:")
    print("Dim: ", hs_data.shape)
    print("Encoding with: ", hs_data.dtype)
    print("Keys:", list(mat_data.keys()))


    print("mean: ", np.mean(hs_data))
    print("Std error: ", np.std(hs_data))
    print("min: ", np.min(hs_data))
    print("max: ", np.max(hs_data))
    print("NaN:", np.any(np.isnan(hs_data)))
    print("Inf: ", np.any(np.isinf(hs_data)))


if __name__ == "__main__":

    # INPUT_DIR = "E:\毕设\Envi_standard_format\WHU-Hi-LongKou"
    # OUTPUT_DIR = "E:\毕设\hyperspect_mat\overall"
    # DATA_SUFFIXES = ['.dat', '.img', '']  # possible addr

    # ENVI2MATConverter.batch_convert(
    #     input_dir=INPUT_DIR,
    #     output_dir=OUTPUT_DIR,
    #     data_suffixes=DATA_SUFFIXES,
    #     save_hdr_info=True
    # )

    root_dir = "/data/chenhaoran/WHUhypspec/data"
    sample_name_list = {"WHU-Hi-LongKou", "WHU-Hi-HongHu", "WHU-Hi-HanChuan"}

    for sample_name in sample_name_list:
        converter_1 = ENVI2MATConverter(
                envi_hdr_path=f"{root_dir}/{sample_name}.hdr",
                envi_data_path=f"{root_dir}/{sample_name}"
    )
        converter_2 = ENVI2MATConverter(
                envi_hdr_path=f"{root_dir}/{sample_name}_gt.hdr",
                envi_data_path=f"{root_dir}/{sample_name}_gt"
    )
        converter_1.convert_to_mat(mat_save_path=f"{root_dir}/{sample_name}.mat")
        converter_2.convert_to_mat(mat_save_path=f"{root_dir}/{sample_name}_gt.mat")

        read_check(sample_name)
