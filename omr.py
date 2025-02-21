import cv2
import numpy as np
import os
import logging
from pdf2image import convert_from_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OMRChecker:
    def __init__(self, image_path):
        self.image_path = image_path
        self.load_image()
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.image.shape[:2]
        
    def load_image(self):
        """Görüntüyü yükle"""
        try:
            self.image = cv2.imread(self.image_path)
            if self.image is None:
                raise Exception("Görüntü yüklenemedi")
            logger.info("Görüntü başarıyla yüklendi")
        except Exception as e:
            logger.error(f"Görüntü yükleme hatası: {str(e)}")
            raise
        
    def preprocess_image(self):
        """Görüntüyü işle"""
        # Görüntüyü yeniden boyutlandır
        scale_percent = 100  # Orijinal boyutu koru
        width = int(self.image.shape[1] * scale_percent / 100)
        height = int(self.image.shape[0] * scale_percent / 100)
        dim = (width, height)
        self.image = cv2.resize(self.image, dim, interpolation=cv2.INTER_AREA)
        
        # Gri tonlamaya çevir
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.image.shape[:2]
        
        # Gürültüyü azalt
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        
        # Adaptif eşikleme uygula
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morfolojik işlemler
        kernel = np.ones((2,2), np.uint8)  # Kernel boyutunu küçült
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Öğrenci ID bölgesi için özel işlem
        student_id_region = thresh[:self.height//3, :]
        student_id_region = cv2.dilate(student_id_region, kernel, iterations=1)
        
        # Birleştir
        thresh[:self.height//3, :] = student_id_region
        
        # Debug için görüntüyü kaydet
        cv2.imwrite('debug_thresh.png', thresh)
        
        return thresh
    
    def find_bubbles(self, thresh):
        """İşaretleme dairelerini bul"""
        # Konturları bul
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        bubbles = []
        min_area = 50  # Minimum daire alanını daha da düşür
        max_area = 800  # Maksimum daire alanını daha da artır
        
        for contour in contours:
            # Kontur alanını hesapla
            area = cv2.contourArea(contour)
            
            # Alan kontrolü
            if min_area < area < max_area:
                # Dairesellik kontrolü
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.5:  # Dairesellik eşiğini daha da düşür
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w/h
                    
                    # Kare benzeri şekiller
                    if 0.6 < aspect_ratio < 1.4:  # Oran aralığını daha da genişlet
                        # Dairenin içindeki piksel değerlerini kontrol et
                        mask = np.zeros(thresh.shape, np.uint8)
                        cv2.drawContours(mask, [contour], -1, 255, -1)
                        mean_val = cv2.mean(self.gray, mask=mask)[0]
                        
                        # Dolu daire kontrolü
                        if mean_val < 220:  # Eşik değerini daha da artır
                            bubbles.append((x, y, w, h))
                            
                            # Debug için görselleştirme
                            cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.drawContours(self.image, [contour], -1, (0, 0, 255), 1)
                            cv2.putText(
                                self.image,
                                f"{mean_val:.0f}",
                                (x, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0),
                                1
                            )
        
        # Debug için görüntüyü kaydet
        cv2.imwrite('debug_contours.png', self.image)
        
        return bubbles
    
    def extract_student_id(self, bubbles):
        """Öğrenci numarasını çıkar"""
        # Üst kısımdaki daireleri seç (öğrenci ID bölgesi)
        student_id_bubbles = [b for b in bubbles if b[1] < self.height/3]
        
        if not student_id_bubbles:
            return ''
            
        # Daireleri y koordinatına göre grupla
        rows = {}
        for bubble in student_id_bubbles:
            x, y, w, h = bubble
            row_idx = int(round(y / (h * 1.2)))
            if row_idx not in rows:
                rows[row_idx] = []
            rows[row_idx].append(bubble)
        
        # Her satırdaki daireleri x koordinatına göre sırala
        for row in rows.values():
            row.sort(key=lambda b: b[0])
        
        # Satırları sırala
        sorted_rows = sorted(rows.keys())
        
        # Her satır için işaretli rakamları bul
        student_id = ''
        for row_idx in sorted_rows:
            row_bubbles = rows[row_idx]
            
            # En koyu daireyi bul
            min_darkness = float('inf')
            marked_bubble = None
            
            for bubble in row_bubbles:
                x, y, w, h = bubble
                roi = self.gray[y:y+h, x:x+w]
                mean_val = cv2.mean(roi)[0]
                
                if mean_val < min_darkness:
                    min_darkness = mean_val
                    marked_bubble = bubble
            
            if marked_bubble:
                # İşaretli dairenin sütun indeksini bul
                col_idx = row_bubbles.index(marked_bubble)
                student_id += str(col_idx)
                
                # Debug için işaretli daireyi göster
                x, y, w, h = marked_bubble
                cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return student_id
    
    def extract_answers(self, bubbles):
        """Cevapları çıkar"""
        # Alt kısımdaki daireleri seç ve sırala
        answer_bubbles = [b for b in bubbles if b[1] > self.height/3]
        answer_bubbles.sort(key=lambda x: (x[1], x[0]))  # Önce y, sonra x'e göre sırala
        
        questions = []
        current_question = []
        last_y = None
        
        for bubble in answer_bubbles:
            x, y, w, h = bubble
            
            if last_y is None or abs(y - last_y) < h*1.5:
                current_question.append(bubble)
            else:
                if current_question:
                    questions.append(current_question)
                current_question = [bubble]
            last_y = y
            
        if current_question:
            questions.append(current_question)
        
        answers = {}
        for i, question in enumerate(questions, 1):
            if len(question) == 5:  # Çoktan seçmeli
                options = ['A', 'B', 'C', 'D', 'E']
            else:  # Doğru/Yanlış
                options = ['T', 'F']
                
            # Daireleri x koordinatına göre sırala
            question.sort(key=lambda x: x[0])
            
            min_darkness = float('inf')
            answer = None
            
            for j, bubble in enumerate(question):
                x, y, w, h = bubble
                roi = self.gray[y:y+h, x:x+w]
                darkness = cv2.mean(roi)[0]
                
                if darkness < min_darkness:
                    min_darkness = darkness
                    answer = options[j]
            
            answers[i] = answer
        
        return answers

    def process(self):
        """Tüm işleme sürecini yönet"""
        try:
            logger.info("Görüntü işleme başladı")
            thresh = self.preprocess_image()
            bubbles = self.find_bubbles(thresh)
            
            if not bubbles:
                raise Exception("Hiç daire bulunamadı")
                
            student_id = self.extract_student_id(bubbles)
            answers = self.extract_answers(bubbles)
            
            results = {
                'student_id': student_id,
                'answers': answers,
                'form_id': 'QL50XB4C'
            }
            
            logger.info(f"İşleme tamamlandı: {results}")
            return results
            
        except Exception as e:
            logger.error(f"İşleme hatası: {str(e)}")
            raise

def main():
    # Test dosya yolu
    image_path = r"C:\Users\Fujitsu\Desktop\paperx\paper-x-client\test.png"
    
    try:
        logger.info(f"Görüntü dosyası okunuyor: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {image_path}")
            
        # Doğrudan görüntüyü oku
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Görüntü yüklenemedi")
            
        # OMRChecker sınıfını başlat
        checker = OMRChecker(image_path)
        results = checker.process()
        
        print("\n" + "="*50)
        print("SINAV SONUÇLARI")
        print("="*50)
        
        # Öğrenci Bilgileri
        print("\nÖĞRENCİ BİLGİLERİ:")
        print("-"*20)
        print(f"Form ID: {results['form_id']}")
        print(f"Öğrenci No: {results['student_id'] if results['student_id'] else 'Tespit edilemedi'}")
        
        # Cevaplar
        print("\nCEVAPLAR:")
        print("-"*20)
        print("\nÇoktan Seçmeli Sorular:")
        for i in range(1, 21):
            answer = results['answers'].get(i)
            print(f"Soru {str(i).zfill(2)}: {answer if answer else 'Boş'}")
        
        print("\nDoğru/Yanlış Soruları:")
        for i in range(21, 31):
            answer = results['answers'].get(i)
            print(f"Soru {str(i).zfill(2)}: {answer if answer else 'Boş'}")
        
        # JSON formatında veri
        print("\nJSON FORMATI:")
        print("-"*20)
        json_data = {
            'student_id': results['student_id'],
            'form_id': results['form_id'],
            'answers': {
                'multiple_choice': {str(i).zfill(2): results['answers'].get(i) for i in range(1, 21)},
                'true_false': {str(i).zfill(2): results['answers'].get(i) for i in range(21, 31)}
            }
        }
        print(json_data)
        
        # İşlenmiş görüntüyü kaydet
        output_dir = os.path.dirname(image_path)
        output_image = os.path.join(output_dir, 'processed_exam_sheet.png')
        cv2.imwrite(output_image, checker.image)
        logger.info(f"İşlenmiş görüntü kaydedildi: {output_image}")
        
        print(f"\nİşlenmiş görüntü kaydedildi: {output_image}")
        print("="*50)
        
    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\nHata: {str(e)}")
    except Exception as e:
        logger.error(f"Program hatası: {str(e)}")
        print(f"\nBeklenmeyen hata: {str(e)}")
        raise

if __name__ == "__main__":
    main() 