# Deprem Yol Analizi Projesi

Bu proje, orijinal Colab Notebook (`Deprem_Yol_Analizi.ipynb`) dosyasının bağımsız bir Python projesine dönüştürülmüş halidir. Veri çekmeden, hazırlığa, model eğitimine ve sokak verileri (OSM) analizine kadar tüm adımlar ayrı betiklere bölünmüştür.

## Kurulum (Gereksinimler)

Gerekli tüm kütüphaneler `requirements.txt` dosyasında listelenmiştir. Yüklemek için terminalden şu komutu çalıştırabilirsiniz:

```bash
pip install -r requirements.txt
```

## Proje Yapısı ve Adımlar

Projeyi kullanırken aşağıdaki sırayı takip edebilirsiniz:

### 1. Veri İndirme (`01_data_download.py`)
Uydu görüntülerini ve maskeleri Hugging Face KATE-CD veri setinden indirir ve `data/images` ve `data/masks` dizinlerine kaydeder.
```bash
python 01_data_download.py
```
*(Not: Zaman kazandırmak için test amaçlı olarak şimdilik ilk 20 resmi indirecek şekilde ayarlıdır. Tüm resimleri çekmek için kodun en sonundaki parametreyi kaldırabilirsiniz.)*

### 2. Veri Hazırlığı (`02_data_preparation.py`)
Segformer modeli için indirilen yüksek çözünürlüklü görüntüleri ağın anlayabileceği 512x512 piksellik parçalara böler (`data/tiles/` dizinine atar).
```bash
python 02_data_preparation.py
```

### 3. Model Eğitimi (`03_train.py`)
Verileri eğitim ve test olarak ayırır, veri artırma (augmentation) uygular ve `segmentation-models-pytorch` kütüphanesini kullanarak SegFormer (mit_b3) modelini eğitir. Başarılı olan en iyi model `models/en_iyi_model.pth` adında kaydedilir.
```bash
python 03_train.py
```

### 4. Çıkarım / Test (Inference) (`04_inference.py`)
Eğitilmiş veya sizin kaydettiğiniz modeli (veya Colab'da eğittiğiniz `.pth` dosyasını `models/` klasörüne taşıyarak) kullanarak rastgele bir resim üzerinde tahmin yürütür ve sonucu görselleştirir.
```bash
python 04_inference.py
```

### 5. OpenStreetMap (OSMNX) Analizi (`05_osmnx_analysis.py`)
Belirtilen bir koordinatın etrafındaki analiz alanındaki sokakları/yolları `osmnx` kullanarak OpenStreetMap sunucularından çeker ve görselleştirir.
```bash
python 05_osmnx_analysis.py
```
