import numpy as np
import os
from pathlib import Path
import json

'''
DEFINING SOME VARIABLES
'''
etfList = ['XLF', 'XLU', 'QQQ', 'SPY', 'XLP', 'EWZ', 'EWH', 'XLY', 'XLE']

# Mevcut veri dizinleri
base_input_dir = './ETF/rectangle/01'
train_data_dir = os.path.join(base_input_dir, 'TrainData')
test_data_dir = os.path.join(base_input_dir, 'TestData')
train_date_dir = os.path.join(base_input_dir, 'Date/TrainDate')
test_date_dir = os.path.join(base_input_dir, 'Date/TestDate')
train_price_dir = os.path.join(base_input_dir, 'Price/TrainPrice')
test_price_dir = os.path.join(base_input_dir, 'Price/TestPrice')

# nnU-Net veri dizini
base_output_dir = './nnUNet_raw/Dataset001_Finance'
imagesTr_dir = os.path.join(base_output_dir, 'imagesTr')
imagesTs_dir = os.path.join(base_output_dir, 'imagesTs')
labelsTr_dir = os.path.join(base_output_dir, 'labelsTr')
labelsTs_dir = os.path.join(base_output_dir, 'labelsTs')

# Klasörleri oluştur
for directory in [imagesTr_dir, imagesTs_dir, labelsTr_dir, labelsTs_dir]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# dataset.json için metadata
dataset_json = {
    "name": "Dataset001_Finance",
    "description": "Financial data for U-Mamba segmentation",
    "tensorImageSize": "2D",
    "modality": {"0": "TechnicalIndicators"},
    "labels": {"0": "Buy", "1": "Hold", "2": "Sell"},
    "numTraining": 0,
    "numTest": 0
}

case_counter = 0  # nnU-Net için dosya isimlendirme

for etf in etfList:
    print(f"\nProcessing {etf}...")

    try:
        # Eğitim verilerini oku
        x_train = np.load(os.path.join(train_data_dir, f'x_{etf}.npy'))
        y_train = np.load(os.path.join(train_data_dir, f'y_{etf}.npy'))
        train_date = np.load(os.path.join(train_date_dir, f'{etf}.npy'), allow_pickle=True)
        train_price = np.load(os.path.join(train_price_dir, f'{etf}.npy'))

        # Test verilerini oku
        x_test = np.load(os.path.join(test_data_dir, f'x_{etf}.npy'))
        y_test = np.load(os.path.join(test_data_dir, f'y_{etf}.npy'))
        test_date = np.load(os.path.join(test_date_dir, f'{etf}.npy'), allow_pickle=True)
        test_price = np.load(os.path.join(test_price_dir, f'{etf}.npy'))

        print(f"Loaded {etf} - Train samples: {len(x_train)}, Test samples: {len(x_test)}")

        # Eğitim verilerini nnU-Net formatına dönüştür
        for idx in range(len(x_train)):
            case_id = f"case_{case_counter:04d}"
            image_path = os.path.join(imagesTr_dir, f"{case_id}_0000.npy")
            label_path = os.path.join(labelsTr_dir, f"{case_id}.npy")

            # Görüntü ve etiketi kaydet
            np.save(image_path, x_train[idx])
            np.save(label_path, y_train[idx])

            case_counter += 1
            dataset_json["numTraining"] += 1

        # Test verilerini nnU-Net formatına dönüştür
        for idx in range(len(x_test)):
            case_id = f"case_{case_counter:04d}"
            image_path = os.path.join(imagesTs_dir, f"{case_id}_0000.npy")
            label_path = os.path.join(labelsTs_dir, f"{case_id}.npy")

            # Görüntü ve etiketi kaydet
            np.save(image_path, x_test[idx])
            np.save(label_path, y_test[idx])

            case_counter += 1
            dataset_json["numTest"] += 1

        # Tarih ve fiyat verilerini kaydet (isteğe bağlı, finansal değerlendirme için)
        for directory in [os.path.join(base_output_dir, 'Date/TrainDate'), os.path.join(base_output_dir, 'Date/TestDate'),
                         os.path.join(base_output_dir, 'Price/TrainPrice'), os.path.join(base_output_dir, 'Price/TestPrice')]:
            Path(directory).mkdir(parents=True, exist_ok=True)

        np.save(f"{base_output_dir}/Date/TrainDate/{etf}.npy", train_date)
        np.save(f"{base_output_dir}/Date/TestDate/{etf}.npy", test_date)
        np.save(f"{base_output_dir}/Price/TrainPrice/{etf}.npy", train_price)
        np.save(f"{base_output_dir}/Price/TestPrice/{etf}.npy", test_price)

    except Exception as e:
        print(f"Error processing {etf}: {str(e)}")
        continue

# dataset.json dosyasını kaydet
with open(os.path.join(base_output_dir, 'dataset.json'), 'w') as f:
    json.dump(dataset_json, f, indent=4)

print(f"Data preparation completed. Total cases: {case_counter}")