import json
import cv2
import numpy as np
from pdf2image import convert_from_path
import os

class OMRProcessor:
    def __init__(self, config_path):
        # Konfigürasyon dosyasını oku
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Debug klasörünü oluştur
        os.makedirs('debug_images', exist_ok=True)
    
    def save_debug_image(self, image, region_config, name):
        """Debug için bölgeyi görüntüye çiz ve kaydet"""
        x1, y1, x2, y2 = region_config['x1'], region_config['y1'], region_config['x2'], region_config['y2']
        
        # Orijinal görüntüyü kopyala
        debug_img = image.copy()
        
        # Bölgeyi dikdörtgen içine al
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Görüntüyü kaydet
        cv2.imwrite(f'debug_images/{name}.png', debug_img)
        
        # Sadece seçili bölgeyi de kaydet
        region = image[y1:y2, x1:x2]
        cv2.imwrite(f'debug_images/{name}_region.png', region)

    def calculate_bubble_positions(self, region_config, question_number, bubble):
        """Kabarcık pozisyonlarını hesapla"""
        x1, y1 = 0, 0  # Kırpılmış görüntüde koordinatlar 0'dan başlar
        bubble_width = region_config['bubble_width']
        bubble_height = region_config['bubble_height']
        
        # Toplam genişlik ve yükseklik
        total_width = region_config['x2'] - region_config['x1']
        total_height = region_config['y2'] - region_config['y1']
        
        # Her soru için yükseklik hesapla (15 soru için)
        question_height = total_height / 15
        
        # Her şık için genişlik hesapla
        option_width = total_width / region_config['bubbles_per_field']
        
        # Baloncuk pozisyonunu hesapla
        # X pozisyonu: Her şık için yatayda eşit aralık
        bx = x1 + (bubble * option_width) + (option_width - bubble_width) // 2
        # Y pozisyonu: Soru numarasına göre dikey pozisyon
        by = y1 + (question_number * question_height) + (question_height - bubble_height) // 2
        
        return int(bx), int(by)

    def visualize_bubble_detection(self, image, region_config, name):
        """Kabarcık tespitini görselleştir"""
        debug_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        
        # Bölgenin tamamını dikdörtgen içine al
        cv2.rectangle(debug_img, 
                     (region_config['x1'], region_config['y1']), 
                     (region_config['x2'], region_config['y2']), 
                     (255, 0, 0), 2)
        
        for field in range(region_config['fields']):
            for bubble in range(region_config['bubbles_per_field']):
                bx, by = self.calculate_bubble_positions(region_config, field, bubble)
                
                bubble_region = image[by:by+region_config['bubble_height'], 
                                   bx:bx+region_config['bubble_width']]
                white_pixels = np.sum(bubble_region == 255)
                total_pixels = region_config['bubble_width'] * region_config['bubble_height']
                fill_ratio = white_pixels / total_pixels
                
                # Kabarcığı çiz
                color = (0, 255, 0) if fill_ratio > region_config['darkness_threshold'] else (0, 0, 255)
                cv2.rectangle(debug_img, (bx, by), 
                            (bx+region_config['bubble_width'], by+region_config['bubble_height']), 
                            color, 2)
                
                # Doluluk oranını ve değeri yaz
                value = region_config['values'][bubble]
                cv2.putText(debug_img, f'{fill_ratio:.2f} ({value})', (bx, by-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(debug_img, f'({bx},{by})', (bx, by+35), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        # Görüntüyü kaydet
        cv2.imwrite(f'debug_images/{name}_bubbles.png', debug_img)

    def enhance_bubble_detection(self, image):
        """Baloncukları daha belirgin hale getir"""
        # Görüntüyü kopyala
        enhanced = image.copy()
        
        # Gürültüyü azalt
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Adaptif eşikleme uygula
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morfolojik işlemler
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Daire tespiti için görüntüyü hazırla
        circles = cv2.HoughCircles(
            thresh,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=20,
            minRadius=int(min(self.config['student_id']['bubble_width'], 
                            self.config['student_id']['bubble_height']) // 2 - 2),
            maxRadius=int(max(self.config['student_id']['bubble_width'], 
                            self.config['student_id']['bubble_height']) // 2 + 2)
        )
        
        # Debug için daireleri çiz
        debug_img = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Dairenin merkezini çiz
                cv2.circle(debug_img, (i[0], i[1]), 2, (0, 255, 0), 3)
                # Dairenin çevresini çiz
                cv2.circle(debug_img, (i[0], i[1]), i[2], (0, 0, 255), 2)
        
        return thresh, debug_img

    def detect_bubble_center(self, bubble_region):
        """Baloncuğun merkez noktasını ve doluluğunu tespit et"""
        # Görüntüyü kopyala
        region = bubble_region.copy()
        
        # Gürültüyü azalt
        blurred = cv2.GaussianBlur(region, (5, 5), 0)
        
        # Eşikleme
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Konturları bul
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, 0
        
        # En büyük konturu al
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Konturun merkezini hesapla
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = bubble_region.shape[1]//2, bubble_region.shape[0]//2
        
        # Doluluğu hesapla (merkez bölgesindeki siyah piksel oranı)
        center_region_size = 5
        x1 = max(0, cx - center_region_size)
        y1 = max(0, cy - center_region_size)
        x2 = min(bubble_region.shape[1], cx + center_region_size)
        y2 = min(bubble_region.shape[0], cy + center_region_size)
        
        center_region = thresh[y1:y2, x1:x2]
        fill_ratio = np.sum(center_region == 255) / center_region.size
        
        return (cx, cy), fill_ratio

    def detect_bubbles(self, image, region_config):
        """Belirli bir bölgedeki kabarcıkları tespit et"""
        results = []
        debug_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        
        # Sabit merkez bölge boyutu
        center_size = 25  # Merkez alan boyutu
        
        # Her soru için (1'den 15'e kadar)
        for question in range(15):
            marked_bubbles = []
            
            # Her şık için kontrol et
            for bubble in range(region_config['bubbles_per_field']):
                bx, by = self.calculate_bubble_positions(region_config, question, bubble)
                # Görüntü sınırlarını kontrol et
                if (by + region_config['bubble_height'] > image.shape[0] or 
                    bx + region_config['bubble_width'] > image.shape[1]):
                    continue
                
                # Baloncuğun merkez noktasını hesapla
                center_x = bx + (region_config['bubble_width'] // 2) - (center_size // 2) 
                center_y = by + (region_config['bubble_height'] // 2) - (center_size // 2)
                
                # Merkez koordinatların sınırlar içinde olduğundan emin ol
                center_x = max(0, min(center_x, image.shape[1] - center_size))
                center_y = max(0, min(center_y, image.shape[0] - center_size))
                
                # Merkez bölgeyi al
                center_region = image[center_y:center_y+center_size, 
                                   center_x:center_x+center_size]
                
                # Merkez bölgedeki beyaz piksel oranını hesapla
                white_pixels = np.sum(center_region == 255)
                total_pixels = center_size * center_size
                fill_ratio = white_pixels / total_pixels
                
                # Debug bilgilerini görüntüye ekle
                color = (0, 255, 0) if fill_ratio > region_config['darkness_threshold'] else (0, 0, 255)
                
                # Tüm baloncuğu çiz
                cv2.rectangle(debug_img, (bx, by), 
                            (bx+region_config['bubble_width'], by+region_config['bubble_height']), 
                            color, 1)
                
                # Merkez bölgeyi belirt
                cv2.rectangle(debug_img, (center_x, center_y), 
                            (center_x+center_size, center_y+center_size), 
                            color, 2)
                
                # Soru numarası ve şık bilgisi
                value = region_config['values'][bubble]
                cv2.putText(debug_img, f'{question+1}', (bx-20, by+10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(debug_img, f'{value}', (bx+5, by+15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(debug_img, f'{fill_ratio:.2f}', (bx+5, by-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                
                # İşaretli baloncukları kaydet
                if fill_ratio > region_config['darkness_threshold']:
                    marked_bubbles.append(value)
            
            # Cevap tipini belirle ve kaydet
            if len(marked_bubbles) == 0:
                results.append('Boş')
            elif len(marked_bubbles) == 1:
                results.append(marked_bubbles[0])  # Tek seçim
            elif len(marked_bubbles) == 2 and set(marked_bubbles) == {'T', 'F'}:
                results.append('Geçersiz T/F')  # Geçersiz True/False
            elif len(marked_bubbles) > 1:
                results.append(f"Multiple:{','.join(sorted(marked_bubbles))}")  # Çoklu seçim
        
        return results, debug_img

    def convert_pdf_to_image(self, pdf_path):
        """PDF'i görüntüye dönüştür"""
        images = convert_from_path(pdf_path)
        return images[0]  # İlk sayfayı al
    
    def find_boundaries(self, thresh_image):
        """Konturları tespit et ve en dış sınırları bul"""
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Minimum alan filtresi uygula
        min_area = 100
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if not valid_contours:
            return None
            
        # Tüm konturları içeren en küçük dikdörtgen
        x_coords = []
        y_coords = []
        for cnt in valid_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_coords.extend([x, x + w])
            y_coords.extend([y, y + h])
            
        if not x_coords or not y_coords:
            return None
            
        # En dış sınırları bul
        left = max(0, min(x_coords) - 10)
        top = max(0, min(y_coords) - 10)
        right = min(thresh_image.shape[1], max(x_coords) + 10)
        bottom = min(thresh_image.shape[0], max(y_coords) + 10)
        
        return (left, top, right, bottom)

    def preprocess_image(self, image):
        """Görüntü ön işleme"""
        # Griye dönüştür (eğer renkli ise)
        if len(np.array(image).shape) == 3:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        else:
            gray = np.array(image)
        
        # Gürültüyü azalt
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptif eşikleme uygula
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morfolojik işlemler
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh, gray

    def auto_crop_region(self, region_img):
        """Bölgeyi otomatik olarak kırp"""
        # Görüntü ön işleme
        if len(np.array(region_img).shape) == 3:
            gray = cv2.cvtColor(np.array(region_img), cv2.COLOR_RGB2GRAY)
        else:
            gray = np.array(region_img)
            
        # Eşikleme uygula
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Konturları bul
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return region_img, (0, 0, region_img.shape[1], region_img.shape[0])
            
        # Tüm konturların sınırlarını al
        all_points = np.vstack([cnt.reshape(-1, 2) for cnt in contours])
        
        # Baloncukları kapsayan en küçük dikdörtgeni belirle
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Kenar boşluğu ekle
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(region_img.shape[1] - x, w + 2*padding)
        h = min(region_img.shape[0] - y, h + 2*padding)
        
        # Görüntüyü kırp
        cropped = region_img[y:y+h, x:x+w]
        
        return cropped, (x, y, x+w, y+h)

    def process_region(self, image, region_config, region_name):
        """Belirli bir bölgeyi işle ve otomatik kırp"""
        # Bölgeyi kes
        x1, y1, x2, y2 = region_config['x1'], region_config['y1'], region_config['x2'], region_config['y2']
        region_img = image[y1:y2, x1:x2]
        
        # Debug için orijinal görüntüyü kaydet
        cv2.imwrite(f'debug_images/{region_name}_original.png', region_img)
        
        # Bölgeyi otomatik kırp
        cropped_img, (left, top, right, bottom) = self.auto_crop_region(region_img)
        
        # Yeni koordinatları güncelle
        new_width = right - left
        new_height = bottom - top
        
        # Baloncuk boyutlarını orantılı olarak güncelle
        scale_x = new_width / (x2 - x1)
        scale_y = new_height / (y2 - y1)
        
        region_config.update({
            'x1': 0,  # Kırpılmış görüntüde koordinatlar 0'dan başlar
            'y1': 0,
            'x2': new_width,
            'y2': new_height,
            'bubble_width': int(region_config['bubble_width'] * scale_x),
            'bubble_height': int(region_config['bubble_height'] * scale_y)
        })
        
        # Görüntüyü işle
        processed_img = cv2.GaussianBlur(cropped_img, (5, 5), 0)
        _, processed_img = cv2.threshold(processed_img, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Debug için görüntüleri kaydet
        cv2.imwrite(f'debug_images/{region_name}_cropped.png', cropped_img)
        cv2.imwrite(f'debug_images/{region_name}_processed.png', processed_img)
        
        return processed_img

    def calculate_student_id_positions(self, region_config, field, bubble):
        """Student ID için kabarcık pozisyonlarını hesapla"""
        x1, y1 = 0, 0  # Kırpılmış görüntüde koordinatlar 0'dan başlar
        bubble_width = region_config['bubble_width']
        bubble_height = region_config['bubble_height']
        
        # Toplam genişlik ve yükseklik
        total_width = region_config['x2'] - region_config['x1']
        total_height = region_config['y2'] - region_config['y1']
        
        # Her sütun için genişlik hesapla
        column_width = total_width / region_config['fields']
        # Her satır için yükseklik hesapla
        row_height = total_height / region_config['bubbles_per_field']
        
        # Baloncuk pozisyonunu hesapla
        # X pozisyonu: Sütun numarasına göre yatay pozisyon
        bx = x1 + (field * column_width) + (column_width - bubble_width) // 2
        # Y pozisyonu: Satır numarasına göre dikey pozisyon
        by = y1 + (bubble * row_height) + (row_height - bubble_height) // 2
        
        return int(bx), int(by)

    def detect_student_id_bubbles(self, image, region_config):
        """Öğrenci ID bölgesindeki kabarcıkları tespit et"""
        results = []
        debug_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        
        # Sabit merkez bölge boyutu
        center_size = 25  # Merkez alan boyutu
        
        # Her sütun için (basamak)
        for field in range(region_config['fields']):
            marked_values = []
            max_fill_ratio = 0
            selected_value = None
            
            # Her satır için (0-9)
            for bubble in range(region_config['bubbles_per_field']):
                bx, by = self.calculate_student_id_positions(region_config, field, bubble)
                
                # Görüntü sınırlarını kontrol et
                if (by + region_config['bubble_height'] > image.shape[0] or 
                    bx + region_config['bubble_width'] > image.shape[1]):
                    continue
                
                # Baloncuğun merkez noktasını hesapla
                center_x = bx + (region_config['bubble_width'] // 2) - (center_size // 2) 
                center_y = by + (region_config['bubble_height'] // 2) - (center_size // 2)
                
                # Merkez koordinatların sınırlar içinde olduğundan emin ol
                center_x = max(0, min(center_x, image.shape[1] - center_size))
                center_y = max(0, min(center_y, image.shape[0] - center_size))
                
                # Merkez bölgeyi al
                center_region = image[center_y:center_y+center_size, 
                                   center_x:center_x+center_size]
                
                # Merkez bölgedeki beyaz piksel oranını hesapla
                white_pixels = np.sum(center_region == 255)
                total_pixels = center_size * center_size
                fill_ratio = white_pixels / total_pixels
                
                # Debug bilgilerini görüntüye ekle
                color = (0, 255, 0) if fill_ratio > region_config['darkness_threshold'] else (0, 0, 255)
                
                # Tüm baloncuğu çiz
                cv2.rectangle(debug_img, (bx, by), 
                            (bx+region_config['bubble_width'], by+region_config['bubble_height']), 
                            color, 1)
                
                # Merkez bölgeyi belirt
                cv2.rectangle(debug_img, (center_x, center_y), 
                            (center_x+center_size, center_y+center_size), 
                            color, 2)
                
                # Değer ve doluluk oranı bilgisi
                value = region_config['values'][bubble]
                cv2.putText(debug_img, f'{value}', (bx+5, by+15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(debug_img, f'{fill_ratio:.2f}', (bx+5, by-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                
                # İşaretli baloncukları ve en yüksek doluluk oranını kaydet
                if fill_ratio > region_config['darkness_threshold']:
                    marked_values.append((value, fill_ratio))
                    if fill_ratio > max_fill_ratio:
                        max_fill_ratio = fill_ratio
                        selected_value = value
            
            # Her basamak için işaretli değeri ekle
            if not marked_values:
                results.append('Boş')
            else:
                # En yüksek doluluk oranına sahip değeri seç
                results.append(selected_value)
        
        return results, debug_img

    def process_student_id(self, gray_image):
        """Öğrenci ID bölgesini işle ve sonuçları döndür"""
        if not self.config.get('student_id'):
            return None

        # Öğrenci ID bölgesini işle
        enhanced_region = self.process_region(gray_image, self.config['student_id'], 'student_id')
        
        # process_region zaten THRESH_BINARY_INV uyguluyor, tekrar threshold yapmaya gerek yok
        student_id, debug_img = self.detect_student_id_bubbles(enhanced_region, self.config['student_id'])
        cv2.imwrite('debug_images/student_id_detection.png', debug_img)
        
        # ID'yi string olarak birleştir (sadece sayısal değerleri al)
        student_id_str = ''.join([str(x) for x in student_id if x != 'Boş'])
        
        return {
            'student_id': student_id_str,
            'debug_image': debug_img,
            'processed_region': enhanced_region
        }

    def process_exam(self, pdf_path):
        """Sınav kağıdını işle ve JSON formatında döndür"""
        # PDF'i görüntüye dönüştür
        image = self.convert_pdf_to_image(pdf_path)
        
        # Görüntü ön işleme
        processed_image, gray_image = self.preprocess_image(image)
        
        # JSON çıktısı için basit sonuç yapısı
        results = {
            'studentId': None,
            'allAnswers': []
        }
        
        # Öğrenci ID'sini işle
        student_id_results = self.process_student_id(gray_image)
        if student_id_results and student_id_results['student_id']:
            results['studentId'] = student_id_results['student_id']
        
        # Cevap bölgelerini işle
        if self.config.get('answer_regions'):
            all_answers = []
            for i, region in enumerate(self.config['answer_regions'], 1):
                enhanced_region = self.process_region(gray_image, region, f'answer_region_{i}')
                answers, debug_img = self.detect_bubbles(enhanced_region, region)
                cv2.imwrite(f'debug_images/answer_region_{i}_detection.png', debug_img)
                
                # Cevapları ana listeye ekle
                all_answers.extend(answers)
            
            # Tüm cevapları tek bir listede topla
            results['allAnswers'] = all_answers
        
        return results

def main():
    # OMR işlemcisini oluştur
    processor = OMRProcessor('omr_coordinates.json')
    
    # Sınav kağıdını işle
    results = processor.process_exam('sinav.pdf')
    
    # JSON formatında yazdır
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
