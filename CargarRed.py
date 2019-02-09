import cv2
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import image
import numpy as np

red = "red_5"
imagen = "mala3.jpeg"

json_file = open( red + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(red + ".h5")
print("Loaded model from disk")

def crop(image,kernel_size,factor=1):
    div = int(kernel_size/2)
    x,y,channels = image.shape
    result = np.zeros((x,y),np.uint8)

    for X in range(div,(x-div),factor):
        for Y in range(div,y-div,factor):

            crop_img = image[X-div:X+div,Y-div:Y+div]
            #cv2.imshow("cropped", crop_img)
            #cv2.waitKey(0)
            crop_img = np.expand_dims(crop_img, axis=0)
            a = loaded_model.predict(crop_img)

            #print(str(X) + "_" +str(Y))
            porc = (X/x)*100
            print("%.2f" % porc)
            result[X][Y] = a[0]*255
    return result

############################################
#Ingrese la imagen con la que desea probar #
############################################

test_img = cv2.imread(imagen)
mascara = crop(test_img,64,1)


test_img2 = cv2.resize(test_img, (64, 64))


test_img2 = np.expand_dims(test_img2, axis=0)
a=loaded_model.predict(test_img2)
print(a)
if a[0] > 0.8:
    resultado = "Mala"
else:
    resultado = "Buena"


######################################
#       mostrar imagen escalada      #
cv2.imshow("foto",test_img)
cv2.waitKey(0)

##############################
#       mostrar mascara     #
cv2.imshow("mascara",mascara)
cv2.waitKey(0)

##########################################
#      resultado con mascara aplicada    #
res = cv2.bitwise_and(test_img,test_img,mask = mascara)
cv2.imshow(resultado,res)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""b=np.uint8((a>0.5))
b=b.T
out=test_img*0
cont2=0
sizex,sizey,sizez = np.shape(out)
print(sizex,sizey,sizez)
#reconstrucción de imagen para evaluar red
for i in range(0,sizex-10):
    for j in range(0,sizey-10):
            out[i+5][j+5][0]= b[0][cont2]
            out[i+5][j+5][1]= b[0][cont2]
            out[i+5][j+5][2]= b[0][cont2]
            cont2=cont2+1
out=out*255
kernel = np.ones((5,5),np.uint8)

#####################################################################
#La configuración mas adecuada es independiente para cada imagen :( #
#####################################################################
dilatacion= cv2.dilate(out.copy(),kernel,iterations =3)
erosion = cv2.erode(dilatacion.copy(),kernel,iterations = 9)
dilatacion1 = cv2.dilate(erosion.copy(),kernel,iterations =7)
"""

#mask = cv2.cvtColor(dilatacion1, cv2.COLOR_BGR2GRAY)
