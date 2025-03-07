import cv2
import numpy as np
import json
from pdf2image import convert_from_path

class RegionSelector:
    def __init__(self):
        self.image = None
        self.display_image = None
        self.scale_factor = 1.0
        self.drawing = False
        self.start_x = -1
        self.start_y = -1
        self.end_x = -1
        self.end_y = -1
        self.regions = {
            'student_id': None,
            'answer_regions': []
        }
        self.current_region = None
        self.window_name = "Bölge Seçici (Yardım için 'h' tuşuna basın)"
        self.temp_rect = None
        self.help_visible = False
        
    def show_help(self):
        """Yardım menüsünü göster"""
        help_img = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(help_img, "YARDIM MENUSU", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        help_text = [
            "1: Ogrenci ID bolgesi secimi",
            "2: Universite ID bolgesi secimi",
            "3: Cevap bolgesi secimi",
            "c: Tum secimleri temizle",
            "s: Secimleri kaydet ve cik",
            "r: Goruntu boyutunu sifirla",
            "+: Yakinlastir",
            "-: Uzaklastir",
            "h: Yardim menusu",
            "ESC: Kaydetmeden cik",
            "",
            "Mouse ile bolge secimi:",
            "1. Ilgili tusu secin (1, 2 veya 3)",
            "2. Sol tik ile secime baslayin",
            "3. Sururkleyerek bolgeyi belirleyin",
            "4. Sol tiki birakarak secimi tamamlayin"
        ]
        
        for i, text in enumerate(help_text):
            cv2.putText(help_img, text, (20, 70 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Yardım", help_img)
    
    def scale_coordinates(self, x, y, reverse=False):
        """Koordinatları ölçeklendir"""
        if reverse:
            return int(x / self.scale_factor), int(y / self.scale_factor)
        return int(x * self.scale_factor), int(y * self.scale_factor)
    
    def mouse_callback(self, event, x, y, flags, param):
        if self.current_region is None:
            return
        
        # Ölçeklendirilmiş koordinatları gerçek koordinatlara dönüştür
        real_x, real_y = self.scale_coordinates(x, y, reverse=True)
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x = real_x
            self.start_y = real_y
            self.end_x = real_x
            self.end_y = real_y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_x = real_x
                self.end_y = real_y
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_x = real_x
            self.end_y = real_y
            
            # Koordinatları düzenle (sol üst ve sağ alt köşe olacak şekilde)
            x1 = min(self.start_x, self.end_x)
            y1 = min(self.start_y, self.end_y)
            x2 = max(self.start_x, self.end_x)
            y2 = max(self.start_y, self.end_y)
            
            if self.current_region == 'answer_regions':
                self.regions['answer_regions'].append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'page': 0,
                    'values': ["A", "B", "C", "D", "E"],
                    'orientation': "vertical",
                    'bubbles_per_field': 5,
                    'fields': 1,
                    'bubble_width': 16,
                    'bubble_height': 16,
                    'darkness_threshold': 0.3
                })
            else:
                self.regions[self.current_region] = {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'page': 0,
                    'values': ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                    'orientation': "vertical",
                    'bubbles_per_field': 10,
                    'fields': 10,
                    'bubble_width': 16,
                    'bubble_height': 16,
                    'darkness_threshold': 0.3
                }
    
    def draw_regions(self, img):
        temp = img.copy()
        
        # Mevcut bölgeleri çiz
        colors = {
            'student_id': (0, 255, 0),  # Yeşil
            'answer_regions': (0, 0, 255)  # Kırmızı
        }
        
        # Öğrenci ID bölgesi
        if self.regions['student_id']:
            r = self.regions['student_id']
            x1, y1 = self.scale_coordinates(r['x1'], r['y1'])
            x2, y2 = self.scale_coordinates(r['x2'], r['y2'])
            cv2.rectangle(temp, (x1, y1), (x2, y2), colors['student_id'], 2)
            cv2.putText(temp, "Ogrenci ID", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['student_id'], 2)

        # Cevap bölgeleri
        for i, r in enumerate(self.regions['answer_regions']):
            x1, y1 = self.scale_coordinates(r['x1'], r['y1'])
            x2, y2 = self.scale_coordinates(r['x2'], r['y2'])
            cv2.rectangle(temp, (x1, y1), (x2, y2), colors['answer_regions'], 2)
            cv2.putText(temp, f"Cevap Bolgesi {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['answer_regions'], 2)
        
        # Şu anda çizilen dikdörtgen
        if self.drawing and self.current_region:
            color = colors[self.current_region]
            x1, y1 = self.scale_coordinates(self.start_x, self.start_y)
            x2, y2 = self.scale_coordinates(self.end_x, self.end_y)
            cv2.rectangle(temp, (x1, y1), (x2, y2), color, 2)
        
        return temp
    
    def adjust_image_size(self, image):
        """Görüntüyü ekran boyutuna göre ayarla"""
        screen_height = 900  # Maksimum yükseklik
        screen_width = 1600   # Maksimum genişlik
        
        height, width = image.shape[:2]
        
        # En-boy oranını koru
        scale_h = screen_height / height
        scale_w = screen_width / width
        self.scale_factor = min(scale_h, scale_w)
        
        if self.scale_factor < 1:
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            return cv2.resize(image, (new_width, new_height))
        
        self.scale_factor = 1.0
        return image
    
    def select_regions(self, pdf_path):
        """Bölge seçimi için ana fonksiyon"""
        # PDF'i görüntüye dönüştür
        images = convert_from_path(pdf_path)
        self.image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
        
        # Görüntüyü ekrana sığacak şekilde ayarla
        self.display_image = self.adjust_image_size(self.image)
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Başlangıçta yardım menüsünü göster
        self.show_help()
        
        while True:
            display_img = self.draw_regions(self.display_image)
            cv2.imshow(self.window_name, display_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Tuş kontrolleri
            if key == ord('1'):
                self.current_region = 'student_id'
                print("Öğrenci ID bölgesini seçin")
            elif key == ord('3'):
                self.current_region = 'answer_regions'
                print("Cevap bölgesini seçin")
            elif key == ord('c'):
                self.regions = {
                    'student_id': None,
                    'answer_regions': []
                }
            elif key == ord('s'):
                self.save_regions()
                break
            elif key == ord('h'):
                self.show_help()
            elif key == ord('r'):
                # Görüntü boyutunu sıfırla
                self.display_image = self.adjust_image_size(self.image)
            elif key == ord('+') or key == ord('='):
                # Yakınlaştır
                self.scale_factor *= 1.1
                self.display_image = cv2.resize(self.image, None, 
                                             fx=self.scale_factor, 
                                             fy=self.scale_factor)
            elif key == ord('-') or key == ord('_'):
                # Uzaklaştır
                self.scale_factor /= 1.1
                self.display_image = cv2.resize(self.image, None, 
                                             fx=self.scale_factor, 
                                             fy=self.scale_factor)
            elif key == 27:  # ESC
                break
        
        cv2.destroyAllWindows()
    
    def save_regions(self):
        """Bölgeleri JSON dosyasına kaydet"""
        # Bölgeleri JSON dosyasına kaydet
        with open('omr_coordinates.json', 'w', encoding='utf-8') as f:
            json.dump(self.regions, f, indent=4, ensure_ascii=False)
        print("Bölgeler kaydedildi!")

def main():
    selector = RegionSelector()
    selector.select_regions('sinav.pdf')

if __name__ == "__main__":
    main() 