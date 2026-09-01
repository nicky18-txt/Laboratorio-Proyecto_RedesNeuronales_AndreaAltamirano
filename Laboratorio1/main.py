import json
import numpy as np
import matplotlib.pyplot as plt

with open('mnist_mlp.json', 'r') as file:
    model = json.load(file)

print("Tipo de model: ", type(model))
print("="*60)
print("1. Cargar e inspeccionar el modelo")
print("="*60)

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
            raise ValueError("Error identificando la funcion")

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

print("="*60)
print("2. Cargar y preparar el conjunto de prueba")
print("="*60)
data = np.load("mnist_test.npz")

images = data['images']
labels = data['labels']
print("\nDatos cargados:")
print("Forma de las imágenes: " + str(images.shape))
print("Forma de las etiquetas: " + str(labels.shape)+"\n")

imagesFloat = images.astype(np.float32)
print(f"Conversion a float: {imagesFloat.dtype}")

print(f"Forma de las imágenes en float: {imagesFloat.shape}\n")

imagesNorm = imagesFloat / model["preprocess"]["scale"]
print(f"Normalizacion (division entre 255): {imagesNorm.shape}\n")

imagesAplanada = imagesNorm.reshape(imagesNorm.shape[0], -1)
print(f"Aplanamiento (forma final): {imagesAplanada.shape}\n")

print(f"Verificacion: {imagesAplanada.shape[0]} x {imagesAplanada.shape[1]} = {imagesAplanada.shape[0]*imagesAplanada.shape[1]} pixeles totales\n")

print("="*60)
print("3. Crear red neuronal")
print("="*60+"\n")
red = red_neuronal(model)
print("Red neuronal creada exitosamente\n")

print("="*60)
print("4. Ejecucion de inferencia sobre el conjunto completo")
print("="*60+"\n")
probab = red.forward(imagesAplanada)
print(f"Forma de probabilidades: {probab.shape}\n")
print("Forward pass completado exitosamente\n")

print("Comprobando salida final. Debe tener la forma  (10000, 10)")
print(f"Forma de salida: {probab.shape}")
if probab.shape == (10000, 10):
    print("La salida final es correcta\n")
else:
    print("La salida final no es la esperada\n")

print("Verificar que cada fila de la salida Softmax sume 1")
sumas = np.sum(probab, axis=1)

if np.allclose(sumas, 1):
    print("Todas las filas suman 1\n")
else:
    print("Algunas filas no suman 1")
    print(f"Suma minima: {sumas.min()}, Suma maxima: {sumas.max()}\n")

print("Clase predicha mediante el indice de mayor probabilidad")
clases_predichas = np.argmax(probab, axis=1)
print(f"Clases predichas (forma): {clases_predichas.shape}")
print(f"Primeras 10 clases predichas: {clases_predichas[:10]}\n")

print("Calcular exactitud (se espera sea de ~96.66%):")
ajustes = np.sum(clases_predichas == labels)
exactitud = ajustes / len(labels)
print(f"Exactitud: {exactitud * 100:.2f}%\n")

print("="*60)
print("5. Predicciones individuales de algunos elementos")
print("="*60+"\n")

for i in range(10):
    label_real = labels[i]
    predic = clases_predichas[i]
    probabilidades = probab[i]
    confianza = probabilidades[predic]

    correcto = "correcto" if label_real == predic else "incorrecto"
    print(f"Imagen {i}: \nEtiqueta real: {label_real} \nPredicción: {predic}, {correcto} \nConfianza: {confianza:.4f} \nProbabilidades:")
    for digito in range(10):
        prob = probabilidades[digito]
        bar = '#' * int(prob * 30)
        print(f"Digito {digito}: {prob:.4f} {bar}")

imagenI = 1

imagenOrg = images[imagenI]
labelR = labels[imagenI]
prediccion = clases_predichas[imagenI]
probabilidades = probab[imagenI]
confianza = probabilidades[prediccion]
print("="*60)
print(f"6. Visualizar imagen")
print("="*60+"\n")
print(f"\nVisualizando la imagen {imagenI}:\n")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(imagenOrg, cmap='gray')
ax1.set_title(f"Imagen Original\nEtiqueta real: {labelR}", fontsize=14, fontweight='bold')
ax1.axis('off')

digit = np.arange(10)
colores = ["blue" if d == prediccion else "gray" for d in digit]
ax2.bar(digit, probabilidades, color=colores, edgecolor='black', linewidth=1.5)
ax2.set_title(f"Prediccions: {prediccion} \nConfianza: {confianza:.4f}", fontsize=14, fontweight='bold')
ax2.set_xlabel("Digito", fontsize=12, fontweight='bold')
ax2.set_ylabel("Probabilidad", fontsize=12, fontweight='bold')

ax2.set_xticks(digit)
ax2.set_ylim([0, 1])
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

