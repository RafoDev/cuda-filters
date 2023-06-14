# CUDA Filters

Se utilizan CUDA C/C++ y stb_image.h para implementar dos filtros (kernels):

- toGreyScale
- blur

## Input

El input para ambos kernels es una imagen, representada como un vector unidimensional de caracteres (0-255) en el que irán los 3 canales de cada píxel. Si la imagen es de 20 x 20, habrán 20 x 20 x 3 elementos en el vector. 
A continuación la función `imageToRGBVector` que convierte una imagen a vector:

```c++
// include/utils.hpp

vector<unsigned char> imageToRGBVector(const string &filename, int &width, int &height, int &channels)
{
	unsigned char *image = stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb);
	vector<unsigned char> rgbVector;

	if (image != nullptr)
	{
		int imageSize = width * height * channels;
		rgbVector.assign(image, image + imageSize);
		stbi_image_free(image);
	}
	return rgbVector;
}
```

Luego el vector resultante tendrá que ser copiado del host a la memoria del dispositivo. 

```c++
// Se utiliza la función imateToRGBVector para obtener el vector de entrada
vector<unsigned char> rgbVectorIn = imageToRGBVector(inputPath, width, height, channels);
// Se crea un puntero a los datos del vector en el host
unsigned char *h_rgbVectorIn = rgbVectorIn.data();
// Se crea un puntero a los datos del vector en el dispositivo
unsigned char *d_rgbVectorIn;
// Se separa memoria para almacenar el vector en el dispositivo
cudaMalloc((void **)&d_rgbVectorIn, size);
// Se copia el vector del host al dispositivo
cudaMemcpy(d_rgbVectorIn, h_rgbVectorIn, size, cudaMemcpyHostToDevice);
```
Las imagenes de entrada se encuentran en la carpeta `images/in`.


## Output

El output de los kernels es un vector unidimensional de caracteres (0-255) modificado por el kernel. Este vector tendrá que ser copiado a la memoria del host.

```c++
// Se crea un puntero a los datos del vector en el host
unsigned char *h_rgbVectorOut = rgbVectorOut.data();
// Se crea un puntero a los datos del vector en el dispositivo
unsigned char *d_rgbVectorOut;
// Se separa memoria para almacenar el vector en el dispositivo
cudaMalloc((void **)&d_rgbVectorOut, size);
// Se copia el vector del host al dispositivo
cudaMemcpy(d_rgbVectorOut, h_rgbVectorOut, size, cudaMemcpyHostToDevice);

// ... procesamiento del kernel sobre d_rgbVectorOut

// Se copia el vector del dispositivo al host
cudaMemcpy(h_rgbVectorOut, d_rgbVectorOut, size, cudaMemcpyDeviceToHost);
```

Posteriormente se utiliza la función `RGBVectorToImage` para convertir el vector en una imagen.

```c++
string outputPath = "../images/out/" + filename;
RGBVectorToImage(rgbVectorOut, width, height, channels, outputPath);
```
Las imagenes de salida se guardan en la carpeta `images/out`.

## toGreyScale

Este kernel realiza una conversión de color a escala de grises en paralelo.

```c++
__global__ void colorToGreyScaleConversion(unsigned char *Pout, unsigned char *Pin, int width, int height)
{
	// Se calculan las coordenadas del píxel correspondiente al hilo en el bloque de hilos 
	int Col = threadIdx.x + blockIdx.x * blockDim.x;
	int Row = threadIdx.y + blockIdx.y * blockDim.y;
	// Se verifica que el píxel se encuentre dentro de los límites de la imagen
	if (Col < width && Row < height)
	{
		// Se calculan los desplazamientos necesarios para acceder al píxel en sus 3 canales
		int greyOffset = Row * width + Col;
		int rgbOffset = greyOffset * CHANNELS;

		// Se obtienen los 3 canales del píxel del vector de entrada
		unsigned char r = Pin[rgbOffset];
		unsigned char g = Pin[rgbOffset + 1];
		unsigned char b = Pin[rgbOffset + 2];

		// Se calcula el valor en la escala de grises del píxel
		unsigned char tmp = 0.21f * r + 0.71f * g + 0.07f * b;

		// Se actualiza el valor del vector de salida
		Pout[rgbOffset] = tmp;
		Pout[rgbOffset + 1] = tmp;
		Pout[rgbOffset + 2] = tmp;
	}
}
```
Ejemplo del escalado aplicado:

<div style="display: flex;">

<img src="https://i.postimg.cc/fbtW6L6y/racoon.jpg" alt="Original" style="width: 300px; height:300px;margin-right: 10px;">

<img src="https://i.postimg.cc/pL7RYvpH/grey-Scaled-racoon.png" alt="GreyScaled" style="width: 300px; height:300px;">

</div>

## blur

Este kernel aplica un efecto de desenfoque a una imagen en paralelo.

El nivel de desenfoque aplicado dependerá del tamaño de la ventana (píxeles vecinos). El tamaño se configura con la variable `BLUR_SIZE`. Para la implementación se consideró un tamaño de 10 con el objetivo de que el desenfoque sea notable. 

```c++
__global__ void blurKernel(unsigned char *Pout, unsigned char *Pin, int width, int height)
{
	// Se calculan las coordenadas del píxel correspondiente al hilo en el bloque de hilos 
	int Col = threadIdx.x + blockIdx.x * blockDim.x;
	int Row = threadIdx.y + blockIdx.y * blockDim.y;

	// Se verifica que el píxel se encuentre dentro de los límites de la imagen
	if (Col < width && Row < height)
	{
		// Se inicializan las variables que almacenaran los valores acumulados de los canales de color de los píxeles vecinos
		int pixVal_r = 0;
		int pixVal_g = 0;
		int pixVal_b = 0;
		int pixels = 0;

		// Se utiliza un bucle anidado para recorrer los píxeles vecinos dentro de una ventana de finida por BLUR_SIZE 
		for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow)
		{
			for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol)
			{
				// Se calcula el desplazamiento para el píxel en la ventana
				int currRow = Row + blurRow;
				int currCol = Col + blurCol;
				
				int rgbOffset = (currRow * width + currCol) * CHANNELS;

				// Se verifica si el píxel se encuentra dentro de los límites de la imagen
				if (currRow > -1 && currRow < height && currCol > -1 && currCol < width)
				{
					// Se acumulan los valores de los canales del píxel dentro de la ventana y se incrementa el contador de píxeles
					pixVal_r += Pin[rgbOffset];
					pixVal_g += Pin[rgbOffset + 1];
					pixVal_b += Pin[rgbOffset + 2];
					pixels++;
				}
			}
		}
		// Se calcula el desplazamiento del píxel del vector de salida
		int blurrOffset = (Row * width + Col) * CHANNELS;
		
		// Los valores promedio se calculan dividiendo las sumas cumulativas por el contador. Los resultados se asignan al vector de salida. 
		Pout[blurrOffset] = (unsigned char)(pixVal_r / pixels);
		Pout[blurrOffset + 1] = (unsigned char)(pixVal_g / pixels);
		Pout[blurrOffset + 2] = (unsigned char)(pixVal_b / pixels);
	}
}
```

Ejemplo del desenfoque aplicado:

<div style="display: flex;">

<img src="https://i.postimg.cc/fbtW6L6y/racoon.jpg" alt="Original" style="width: 300px; height:300px; margin-right: 10px;">

<img src="https://i.postimg.cc/9FLW0Z2z/blurred-racoon.png" alt="Blurred" style="width: 300px; height:300px;">

</div>


## Configuración de lanzamiento
La configuración de lanzamiento de los kernels consistirá en 2 variables: `dimGrid` y `dimBloc`
```c++
dim3 dimGrid(ceil(width / 16.0), ceil(height / 16.0), 1);
dim3 dimBlock(16, 16, 1);

Kernel<<<dimGrid, dimBlock>>>(d_rgbVectorOut, d_rgbVectorIn, width, height);
```
La variable `dimGrid` define las dimensiones del grid. Se utiliza la función `ceil` para redondear hacia arriba la división del ancho y alto de la imagen entre 16.0 (tamaño del bloque). Esto asegura que hayan suficientes bloques para cubrir todos los píxeles de la imagen. 

La variable `dimBloc` define las dimensiones del bloque de hilos. Se establece un bloque de 16x16x1, lo que significa que habrán 256 hilos por bloque.

## Compilación y uso

Para clonar el proyecto:

```bash
git clone https://github.com/RafoDev/cuda-filters.git
```
Para compilarlo:

```bash
cmake -B build
cd build
make
```
Se obtendrán dos binarios:

- toGreyScale
- blur

Para utilizarlos:

```bash
./toGreyScale racoon.jpg
./blur blur.jpg
```

Basta con pasar como parámetro el nombre de la imagen dentro de la carpeta `images/out`. Automáticamente se creará una imagen de salida en la carpeta `images/out` con el mismo nombre, pero anteponiendo el nombre del filtro aplicado, por ejemplo `blurred-racoon.jpg`.

