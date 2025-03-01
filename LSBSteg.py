import cv2
import docopt
import numpy as np
class ImageWrapper:
    def __init__(self, img):
        self.img = img
        self.height, self.width = img.shape[:2]
        self.channels = 3  # Forziamo a 3 canali per immagini RGB
   
    def __getitem__(self, index):
        return self.img[index]
    
    def __setitem__(self, index, value):
        self.img[index] = value

    @property
    def array(self):
        return self.img

class SteganographyException(Exception):
    pass


class LSBSteg():
    def __init__(self, im):
        self.image = im
        self.height, self.width, self.nbchannels = im.shape
        self.size = self.width * self.height
        
        self.maskONEValues = [1,2,4,8,16,32,64,128]
        #Mask used to put one ex:1->00000001, 2->00000010 .. associated with OR bitwise
        self.maskONE = self.maskONEValues.pop(0) #Will be used to do bitwise operations
        
        self.maskZEROValues = [254,253,251,247,239,223,191,127]
        #Mak used to put zero ex:254->11111110, 253->11111101 .. associated with AND bitwise
        self.maskZERO = self.maskZEROValues.pop(0)
        
        self.curwidth = 0  # Current width position
        self.curheight = 0 # Current height position
        self.curchan = 0   # Current channel position

    def put_binary_value(self, bits): #Put the bits in the image
        for c in bits:
            val = list(self.image[self.curheight,self.curwidth]) #Get the pixel value as a list
            if int(c) == 1:
                val[self.curchan] = int(val[self.curchan]) | self.maskONE #OR with maskONE
            else:
                val[self.curchan] = int(val[self.curchan]) & self.maskZERO #AND with maskZERO
                
            self.image[self.curheight,self.curwidth] = tuple(val)
            self.next_slot() #Move "cursor" to the next space
        
    def next_slot(self):#Move to the next slot were information can be taken or put
        if self.curchan == self.nbchannels-1: #Next Space is the following channel
            self.curchan = 0
            if self.curwidth == self.width-1: #Or the first channel of the next pixel of the same line
                self.curwidth = 0
                if self.curheight == self.height-1:#Or the first channel of the first pixel of the next line
                    self.curheight = 0
                    if self.maskONE == 128: #Mask 1000000, so the last mask
                        raise SteganographyException("No available slot remaining (image filled)")
                    else: #Or instead of using the first bit start using the second and so on..
                        self.maskONE = self.maskONEValues.pop(0)
                        self.maskZERO = self.maskZEROValues.pop(0)
                else:
                    self.curheight +=1
            else:
                self.curwidth +=1
        else:
            self.curchan +=1

    def read_bit(self): #Read a single bit int the image
        val = self.image[self.curheight,self.curwidth][self.curchan]
        val = int(val) & self.maskONE
        self.next_slot()
        if val > 0:
            return "1"
        else:
            return "0"
    
    def read_byte(self):
        return self.read_bits(8)
    
    def read_bits(self, nb): #Read the given number of bits
        bits = ""
        for i in range(nb):
            bits += self.read_bit()
        return bits

    def byteValue(self, val):
        return self.binary_value(val, 8)
        
    def binary_value(self, val, bitsize): #Return the binary value of an int as a byte
        binval = bin(val)[2:]
        if len(binval) > bitsize:
            raise SteganographyException("binary value larger than the expected size")
        while len(binval) < bitsize:
            binval = "0"+binval
        return binval

    def encode_text(self, txt):
        l = len(txt)
        binl = self.binary_value(l, 16) 
        self.put_binary_value(binl) 
        for char in txt: 
            c = ord(char)
            self.put_binary_value(self.byteValue(c))
        return self.image
       
    def decode_text(self):
        ls = self.read_bits(16) 
        l = int(ls,2)
        i = 0
        unhideTxt = ""
        while i < l: 
            tmp = self.read_byte()
            i += 1
            unhideTxt += chr(int(tmp,2)) 
        return unhideTxt

    def encode_image(self, imtohide):
        w = imtohide.width
        h = imtohide.height
        if self.width*self.height*self.nbchannels < w*h*imtohide.channels:
            raise SteganographyException("Carrier image not big enough to hold all the datas to steganography")
        binw = self.binary_value(w, 16)
        binh = self.binary_value(h, 16)
        self.put_binary_value(binw) 
        self.put_binary_value(binh) 
        for h in range(imtohide.height): 
            for w in range(imtohide.width):
                for chan in range(imtohide.channels):
                    val = imtohide[h,w][chan]
                    self.put_binary_value(self.byteValue(int(val)))
        return self.image

    
    def decode_image(self) :
        width = int(self.read_bits(16), 2)
        height = int(self.read_bits(16), 2)
        unhideimg = np.zeros((height, width, 3), np.uint8)
        wrapped_unhideimg = ImageWrapper(unhideimg)
        for h in range(height):
            for w in range(width):
                for chan in range(3):  # Usiamo direttamente 3 per i canali RGB
                    val = int(self.read_byte(), 2)
                    unhideimg[h, w, chan] = val
        return unhideimg
    def encode_binary(self, data):
        l = len(data)
        if self.width*self.height*self.nbchannels < l+64:
            raise SteganographyException("Carrier image not big enough to hold all the datas to steganography")
        self.put_binary_value(self.binary_value(l, 64))
        for byte in data:
            byte = byte if isinstance(byte, int) else ord(byte) # Compat py2/py3
            self.put_binary_value(self.byteValue(byte))
        return self.image
    
    def decode_binary(self):
        l = int(self.read_bits(64), 2)
        output = b""
        for i in range(l):
            output += bytearray([int(self.read_byte(),2)])
        return output
    def gaussian_noisy(self, mean, sigma): #mean = media, sigma=dev.std. 
        x1, y1, z1 = self.shape
        gauss_noise = np.random.normal(mean, sigma, (x1, y1, z1)).astype(np.float32)
        noisy_image = self.astype(np.float32) + gauss_noise
        gn_img = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return gn_img

        """
            un appunto sulla funzione gaussiana:
                sigma:
                    10-25 -> rumore leggero
                    25-50 -> rumore medio
                    50-100 -> per un rumore più aggressivo
                mean:
                    di solito è un valore vicino lo "0" -> poiché si sposta troppo potrebbero esserci dei cambiamenti significativi
                    in modo tale che non tende né a scurire né a chiarire l'immagine.
            
            
        """



