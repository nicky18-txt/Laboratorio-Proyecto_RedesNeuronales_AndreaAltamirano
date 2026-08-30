import json
import numpy as np


with open('mnist_mlp.json', 'r') as file:
    model = json.load(file)

print("Campos del modelo:" + str(model.keys()))

capas = model['layers']

for layer in range(len(capas)):
    print("Capa " + str(layer + 1) + ": ")

    num_capa = capas[layer]
    print("Tipo de la capa: " + num_capa['type'])
    print("Unidades: " + str(num_capa['units']))
    print("Funcion de activacion:" + num_capa['activation'])
    print(f"Forma de W: {len(num_capa['W'])}, {len(num_capa['W'][0])}")
    print(f"Forma de b: {len(num_capa['b'])}")
    print("\n")
    
    W_array = np.array(num_capa['W'])
    b_array = np.array(num_capa['b'])
    print("\n")
    #print(f"Forma de W_array: {W_array.shape}")
    #print(f"Forma de b_array: {b_array.shape}")

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class capa_densa:

    def __init__(self, weights, biases, activation):
        self.weights = weights
        self.activation = activation
        self.biases = biases

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.biases

        if self.activation == 'relu':
            return relu(z)
        elif self.activation == 'softmax':
            return softmax(z)
        else:
            raise ValueError(f"Función de activación desconocida: {self.activation}")

class red_neuronal:

    def __init__(self, model):
        self.layers = []

        for layer in model['layers']:
            weights = np.array(layer['W'])
            biases = np.array(layer['b'])
            activation = layer['activation']
            self.layers.append(capa_densa(weights, biases, activation))

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output


data = np.load("mnist_test.npz")

images = data['images']
labels = data['labels']

print("Forma de las imágenes: " + str(images.shape))
print("Forma de las etiquetas: " + str(labels.shape))

imagesFloat = images.astype(np.float32)
print(f"Conversion a float: {imagesFloat.dtype}")

print(f"Conversion a float: {imagesFloat.shape}")

imagesNorm = imagesFloat / model["preprocess"]["scale"]
print(f"Normalizacion: {imagesNorm.shape}")

imagesAplanada = imagesNorm.reshape(imagesNorm.shape[0], -1)
print(f"Aplanamiento: {imagesAplanada.shape}")

print(f"Verificacion: {imagesAplanada.shape[0]} x {imagesAplanada.shape[1]} = {imagesAplanada.shape[0]*imagesAplanada.shape[1]} pixeles totales")

print("Crear red neuronal")
red = red_neuronal(model)
print("Red neuronal creada exitosamente")

print("Forward pass de la red neuronal")
predicciones = red.forward(imagesAplanada)
print("Forward pass completado exitosamente")

print("Comprobando salida final. Debe tener la forma  (10000, 10)")
print(f"Forma de las predicciones: {predicciones.shape}")
if predicciones.shape == (10000, 10):
    print("La salida final es correcta")
else:
    print("La salida final no es la esperada")

print("Verificar que cada fila de la salida Softmax sume 1")
sumas = np.sum(predicciones, axis=1)

if np.allclose(sumas, 1):
    print("Todas las filas suman 1")
else:
    print("Algunas filas no suman 1")
    print(f"Suma minima: {sumas.min()}, Suma maxima: {sumas.max()}")

print("Clase predicha mediante el indice de mayor probabilidad")
clases_predichas = np.argmax(predicciones, axis=1)
print(f"Clases predichas: {clases_predichas.shape}")
print(f"Primeras 10 clases predichas: {clases_predichas[:10]}")

print("Calcular exactitud (se espera sea de ~96.66%):")
ajustes = np.sum(clases_predichas == labels)
exactitud = ajustes / len(labels)
print(f"Exactitud: {exactitud * 100:.2f}%")


print("Predicciones individuales de algunos elementos")

for i in range(10):
    label_real = labels[i]
    predic = clases_predichas[i]
    probabilidades = predicciones[i]
    confianza = probabilidades[predic]

    correcto = "correcto" if label_real == predic else "incorrecto"
    print(f"Imagen {i}: Etiqueta real: {label_real}, Predicción: {predic}, {correcto}, Confianza: {confianza:.4f} , Probabilidades:")
    for digito in range(10):
        prob = probabilidades[digito]
        bar = '#' * int(prob * 30)
        print(f"Digito {digito}: {prob:.4f} {bar}")