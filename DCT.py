import os
import shutil
import cv2
import sys
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy import signal
quant = np.array([[16,11,10,16,24,40,51,61],      # QUANTIZATION TABLE
                    [12,12,14,19,26,58,60,55],    # required for DCT
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])
'''def show(im):
    im_resized = cv2.resize(im, (500, 500), interpolation=cv2.INTER_LINEAR)
    plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
    plt.show()'''

'''
class DWT():   
    #encoding part : 
    def encode_image(self,img,secret_msg):
        #show(img)
        #get size of image in pixels
        row,col = img.shape[:2]
        #addPad
        if row%8 != 0 or col%8 != 0:
            img = cv2.resize(img,(col+(8-col%8),row+(8-row%8)))
        bImg,gImg,rImg = cv2.split(img)
        bImg = self.iwt2(bImg)
        #get size of paddded image in pixels
        height,width = bImg.shape[:2]
        index = 0
        for row in range(height):
            for col in range(width):
                if img.mode != 'RGB':
                    r, g, b ,a = img.getpixel((col, row))
                elif img.mode == 'RGB':
                    r, g, b = img.getpixel((col, row))
                # first value is length of msg
                if row == 0 and col == 0 and index < length:
                    asc = length
                elif index <= length:
                    c = msg[index -1]
                    asc = ord(c)
                else:
                    asc = r
                encoded.putpixel((col, row), (asc, g , b))
                index += 1


        return sImg

    #decoding part :
    def decode_image(self,img):
        msg = ""
        #get size of image in pixels
        row,col = img.shape[:2]
        bImg,gImg,rImg = cv2.split(img)

        return msg
      
    """Helper function to 'stitch' new image back together"""
    def _iwt(self,array):
        output = np.zeros_like(array)
        nx, ny = array.shape
        x = nx // 2
        for j in xrange(ny):
            output[0:x,j] = (array[0::2,j] + array[1::2,j])//2
            output[x:nx,j] = array[0::2,j] - array[1::2,j]
        return output

    def _iiwt(self,array):
        output = np.zeros_like(array)
        nx, ny = array.shape
        x = nx // 2
        for j in xrange(ny):
            output[0::2,j] = array[0:x,j] + (array[x:nx,j] + 1)//2
            output[1::2,j] = output[0::2,j] - array[x:nx,j]
        return output

    def iwt2(self,array):
        return _iwt(_iwt(array.astype(int)).T).T

    def iiwt2(self,array):
        return _iiwt(_iiwt(array.astype(int).T).T)


        '''

class DCT():    
    def __init__(self): # Constructor
        self.message = None
        self.bitMess = None
        self.oriCol = 0
        self.oriRow = 0
        self.numBits = 0   
    #encoding part : 
    def encode_image(self,img,secret_msg):
        #show(img)
        secret=secret_msg
        self.message = str(len(secret))+'*'+secret
        self.bitMess = self.toBits()
        #get size of image in pixels
        row,col = img.shape[:2]
        ##col, row = img.size
        self.oriRow, self.oriCol = row, col  
        if((col/8)*(row/8)<len(secret)):
            print("Error: Message too large to encode in image")
            return False
        #make divisible by 8x8
        if row%8 != 0 or col%8 != 0:
            img = self.addPadd(img, row, col)
        
        row,col = img.shape[:2]
        ##col, row = img.size
        #split image into RGB channels
        bImg,gImg,rImg = cv2.split(img)
        #message to be hid in blue channel so converted to type float32 for dct function
        bImg = np.float32(bImg)
        #break into 8x8 blocks
        imgBlocks = [np.round(bImg[j:j+8, i:i+8]-128) for (j,i) in itertools.product(range(0,row,8),
                                                                       range(0,col,8))]
        #Blocks are run through DCT function
        dctBlocks = [np.round(cv2.dct(img_Block)) for img_Block in imgBlocks]
        #blocks then run through quantization table
        quantizedDCT = [np.round(dct_Block/quant) for dct_Block in dctBlocks]
        #set LSB in DC value corresponding bit of message
        messIndex = 0
        letterIndex = 0
        for quantizedBlock in quantizedDCT:
            #find LSB in DC coeff and replace with message bit
            DC = quantizedBlock[0][0]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            DC[7] = self.bitMess[messIndex][letterIndex]
            DC = np.packbits(DC)
            DC = np.float32(DC)
            DC= DC-255
            quantizedBlock[0][0] = DC
            letterIndex = letterIndex+1
            if letterIndex == 8:
                letterIndex = 0
                messIndex = messIndex + 1
                if messIndex == len(self.message):
                    break
        #blocks run inversely through quantization table
        sImgBlocks = [quantizedBlock *quant+128 for quantizedBlock in quantizedDCT]
        #blocks run through inverse DCT
        #sImgBlocks = [cv2.idct(B)+128 for B in quantizedDCT]
        #puts the new image back together
        sImg=[]
        for chunkRowBlocks in self.chunks(sImgBlocks, col/8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)
        #converted from type float32
        sImg = np.uint8(sImg)
        #show(sImg)
        sImg = cv2.merge((sImg,gImg,rImg))
        return sImg

    #decoding part :
    def decode_image(self,img):
        row,col = img.shape[:2]
        messSize = None
        messageBits = []
        buff = 0
        #split image into RGB channels
        bImg,gImg,rImg = cv2.split(img)
         #message hid in blue channel so converted to type float32 for dct function
        bImg = np.float32(bImg)
        #break into 8x8 blocks
        imgBlocks = [bImg[j:j+8, i:i+8]-128 for (j,i) in itertools.product(range(0,row,8),
                                                                       range(0,col,8))]    
        #blocks run through quantization table
        #quantizedDCT = [dct_Block/ (quant) for dct_Block in dctBlocks]
        quantizedDCT = [img_Block/quant for img_Block in imgBlocks]
        i=0
        #message extracted from LSB of DC coeff
        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][0]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            if DC[7] == 1:
                buff+=(0 & 1) << (7-i)
            elif DC[7] == 0:
                buff+=(1&1) << (7-i)
            i=1+i
            if i == 8:
                messageBits.append(chr(buff))
                buff = 0
                i =0
                if messageBits[-1] == '*' and messSize is None:
                    try:
                        messSize = int(''.join(messageBits[:-1]))
                    except:
                        pass
            if len(messageBits) - len(str(messSize)) - 1 == messSize:
                return ''.join(messageBits)[len(str(messSize))+1:]
        #blocks run inversely through quantization table
        sImgBlocks = [quantizedBlock *quant+128 for quantizedBlock in quantizedDCT]
        #blocks run through inverse DCT
        #sImgBlocks = [cv2.idct(B)+128 for B in quantizedDCT]
        #puts the new image back together
        sImg=[]
        for chunkRowBlocks in self.chunks(sImgBlocks, col/8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)
        #converted from type float32
        sImg = np.uint8(sImg)
        sImg = cv2.merge((sImg,gImg,rImg))
        ##sImg.save(img)
        #dct_decoded_image_file = "dct_" + original_image_file
        #cv2.imwrite(dct_decoded_image_file,sImg)
        return ''
      
    """Helper function to 'stitch' new image back together"""
    def chunks(self, l, n):
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]
    def addPadd(self,img, row, col):
        img = cv2.resize(img,(col+(8-col%8),row+(8-row%8)))    
        return img
    def toBits(self):
        bits = []
        for char in self.message:
            binval = bin(ord(char))[2:].rjust(8,'0')
            bits.append(binval)
        self.numBits = bin(len(bits))[2:].rjust(8,'0')
        return bits
    #parte 2:
    
    def encode_image_with_image(self, host_img, target_img):
        """
        Encode target image into the host image using DCT coefficients.
        Args:
            host_img (numpy array): The host image where the target image will be embedded.
            target_img (numpy array): The image to embed into the host image.
        Returns:
            numpy array: The encoded image with the target image hidden.
        """
        # Resize target image to fit into the host
        h_row, h_col = host_img.shape[:2]
        t_row, t_col = target_img.shape[:2]
        
        if t_row > h_row or t_col > h_col:
            print("Error: Target image is larger than host image.")
            return None
        
        # Pad target image to match host dimensions
        target_img_resized = cv2.resize(target_img, (h_col, h_row))
        
        # Convert host and target images to float32 for DCT processing
        host_img = np.float32(host_img)
        target_img_resized = np.float32(target_img_resized)
        
        # Split channels
        host_b, host_g, host_r = cv2.split(host_img)
        target_b, target_g, target_r = cv2.split(target_img_resized)
        
        # Encode each channel
        encoded_b = self._encode_channel(host_b, target_b)
        encoded_g = self._encode_channel(host_g, target_g)
        encoded_r = self._encode_channel(host_r, target_r)
        
        # Merge encoded channels
        encoded_img = cv2.merge((encoded_b, encoded_g, encoded_r))
        return np.uint8(encoded_img)

    def decode_image_with_image(self, encoded_img, original_host_img):
        """
        Decode the hidden image from an encoded host image.
        Args:
            encoded_img (numpy array): The image with the hidden target image.
            original_host_img (numpy array): The original host image for decoding.
        Returns:
            numpy array: The extracted hidden image.
        """
        # Convert images to float32
        encoded_img = np.float32(encoded_img)
        original_host_img = np.float32(original_host_img)
        
        # Split channels
        encoded_b, encoded_g, encoded_r = cv2.split(encoded_img)
        host_b, host_g, host_r = cv2.split(original_host_img)
        
        # Decode each channel
        decoded_b = self._decode_channel(encoded_b, host_b)
        decoded_g = self._decode_channel(encoded_g, host_g)
        decoded_r = self._decode_channel(encoded_r, host_r)
        
        # Merge decoded channels
        decoded_img = cv2.merge((decoded_b, decoded_g, decoded_r))
        return np.uint8(decoded_img)

    def _encode_channel(self, host_channel, target_channel):
        """
        Encode a single channel of the target image into the host image using DCT coefficients.
        """
        # Break into 8x8 blocks
        h_blocks = self._get_blocks(host_channel)
        t_blocks = self._get_blocks(target_channel)
        
        # Modify host DCT blocks with target DCT blocks
        encoded_blocks = []
        for h_block, t_block in zip(h_blocks, t_blocks):
            h_dct = cv2.dct(h_block)
            t_dct = cv2.dct(t_block)
            # Embed low-frequency coefficients of target into host
            h_dct[:4, :4] += t_dct[:4, :4] * 0.1  # Use a small weight for embedding
            encoded_blocks.append(cv2.idct(h_dct))
        
        # Reconstruct the channel
        return self._reconstruct_from_blocks(encoded_blocks, host_channel.shape)

    def _decode_channel(self, encoded_channel, host_channel):
        """
        Decode a single channel of the target image from the encoded host image.
        """
        # Break into 8x8 blocks
        e_blocks = self._get_blocks(encoded_channel)
        h_blocks = self._get_blocks(host_channel)
        
        # Extract embedded DCT coefficients
        decoded_blocks = []
        for e_block, h_block in zip(e_blocks, h_blocks):
            e_dct = cv2.dct(e_block)
            h_dct = cv2.dct(h_block)
            # Extract low-frequency coefficients of target
            t_dct = (e_dct[:4, :4] - h_dct[:4, :4]) / 0.1
            t_block = cv2.idct(t_dct)
            decoded_blocks.append(t_block)
        
        # Reconstruct the channel
        return self._reconstruct_from_blocks(decoded_blocks, host_channel.shape)

    def _get_blocks(self, channel):
        """
        Divide the image channel into 8x8 blocks.
        """
        row, col = channel.shape
        blocks = [channel[j:j+8, i:i+8] for j in range(0, row, 8) for i in range(0, col, 8)]
        return [np.float32(block) - 128 for block in blocks]  # Center pixel values around 0

    def _reconstruct_from_blocks(self, blocks, shape):
        """
        Reconstruct an image channel from 8x8 blocks.
        """
        row, col = shape
        reconstructed = np.zeros((row, col), dtype=np.float32)
        block_idx = 0
        for j in range(0, row, 8):
            for i in range(0, col, 8):
                reconstructed[j:j+8, i:i+8] = blocks[block_idx] + 128  # Re-center pixel values
                block_idx += 1
        return reconstructed

class Compare():
    def correlation(self, img1, img2):
        return signal.correlate2d (img1, img2)
    def meanSquareError(self, img1, img2):
        error = np.sum((img1.astype('float') - img2.astype('float')) ** 2)
        error /= float(img1.shape[0] * img1.shape[1]);
        return error
    def psnr(self, img1, img2):
        mse = self.meanSquareError(img1,img2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))