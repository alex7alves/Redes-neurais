
/*
    Implementação da rede neural MLP

    Autor: Alex Alves

*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

void Alocar(int **x)
{
    *x = (int*)malloc(sizeof(int));
}

void AlocarMatriz(double ***mat,int *l,int *c)
{
    int i;
    *mat =(double**)malloc(*l*sizeof(double*));

    for(i=0;i<*l;i++){
        (*mat)[i]= (double*)malloc(*c*sizeof(double));
    }
}


void tamanho(int *l, int *c)
{

    printf("Digite o numero de linhas \n");
    scanf("%d",l);
    printf("Digite o numero de colunas \n");
    scanf("%d",c);
}

int main()
{
    int i, maximo;
    int *NumeroEntradas,*NumeroAamostras;
    double **x, **d;
    double n,a, epsilon;

    n=0.8;
    a=0.001;
    epsilon= 0.000000001;

    Alocar(&NumeroEntradas);
    Alocar(&NumeroAamostras);

    printf(" Para a matriz x: \n\n");
    tamanho(NumeroEntradas,NumeroAamostras);
    AlocarMatriz(&x,NumeroEntradas,NumeroAamostras);
    return 0;
}
