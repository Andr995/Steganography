class DCT_Image():    
    def __init__(self):
        self.oriCol = 0
        self.oriRow = 0
        
    def prepare_hidden_image(self, hidden_img):
        # Converti l'immagine da nascondere in scala di grigi e ridimensionala
        if len(hidden_img.shape) > 2:
            hidden_img = cv2.cvtColor(hidden_img, cv2.COLOR_BGR2GRAY)
        
        # Normalizza i valori tra 0 e 1
        hidden_img = hidden_img.astype(np.float32) / 255.0
        return hidden_img
        
    def encode_image(self, cover_img, hidden_img):
        # Prepara l'immagine da nascondere
        hidden_img = self.prepare_hidden_image(hidden_img)
        hidden_height, hidden_width = hidden_img.shape
        
        # Ottieni dimensioni dell'immagine di copertura
        row, col = cover_img.shape[:2]
        self.oriRow, self.oriCol = row, col
        
        # Verifica se l'immagine può contenere l'immagine nascosta
        if (col/8) * (row/8) < hidden_width * hidden_height:
            print("Error: Hidden image too large to encode in cover image")
            return False
            
        # Aggiungi padding se necessario
        if row%8 != 0 or col%8 != 0:
            cover_img = self.addPadd(cover_img, row, col)
        
        row, col = cover_img.shape[:2]
        
        # Separa i canali
        bImg, gImg, rImg = cv2.split(cover_img)
        bImg = np.float32(bImg)
        
        # Dividi in blocchi 8x8
        imgBlocks = [np.round(bImg[j:j+8, i:i+8]-128) for (j,i) in itertools.product(range(0,row,8),
                                                                       range(0,col,8))]
        
        # Applica DCT ai blocchi
        dctBlocks = [np.round(cv2.dct(img_Block)) for img_Block in imgBlocks]
        
        # Quantizza i blocchi DCT
        quantizedDCT = [np.round(dct_Block/quant) for dct_Block in dctBlocks]
        
        # Inserisci i pixel dell'immagine nascosta nei coefficienti DCT
        hidden_pixels = hidden_img.flatten()
        for idx, quantizedBlock in enumerate(quantizedDCT):
            if idx < len(hidden_pixels):
                # Modifica il coefficiente AC(1,1) invece del DC
                # Normalizza e scala il pixel nascosto
                hidden_val = int(hidden_pixels[idx] * 100)  # Scala per maggiore precisione
                quantizedBlock[1][1] = hidden_val
        
        # Ricostruisci l'immagine
        sImgBlocks = [quantizedBlock * quant + 128 for quantizedBlock in quantizedDCT]
        
        # Ricomponi l'immagine
        sImg = []
        for chunkRowBlocks in self.chunks(sImgBlocks, col/8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
                    
        sImg = np.array(sImg).reshape(row, col)
        sImg = np.uint8(sImg)
        final_img = cv2.merge((sImg, gImg, rImg))
        
        return final_img, hidden_height, hidden_width
        
    def decode_image(self, stego_img, hidden_height, hidden_width):
        row, col = stego_img.shape[:2]
        
        # Separa i canali
        bImg, gImg, rImg = cv2.split(stego_img)
        bImg = np.float32(bImg)
        
        # Dividi in blocchi 8x8
        imgBlocks = [bImg[j:j+8, i:i+8]-128 for (j,i) in itertools.product(range(0,row,8),
                                                                       range(0,col,8))]
        
        # Quantizza i blocchi
        quantizedDCT = [img_Block/quant for img_Block in imgBlocks]
        
        # Estrai i pixel nascosti
        hidden_pixels = []
        total_pixels = hidden_height * hidden_width
        
        for idx, quantizedBlock in enumerate(quantizedDCT):
            if idx < total_pixels:
                # Estrai il valore dal coefficiente AC(1,1)
                hidden_val = quantizedBlock[1][1]
                # Denormalizza il valore
                hidden_pixels.append(hidden_val / 100)  # Descala il valore
        
        # Ricostruisci l'immagine nascosta
        hidden_img = np.array(hidden_pixels[:total_pixels]).reshape(hidden_height, hidden_width)
        
        # Normalizza i valori tra 0 e 255
        hidden_img = np.uint8(np.clip(hidden_img * 255, 0, 255))
        
        return hidden_img
        
    def chunks(self, l, n):
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]
            
    def addPadd(self, img, row, col):
        img = cv2.resize(img,(col+(8-col%8),row+(8-row%8)))    
        return img
