
% Implementação de uma MLP para uma porta XOR e uma porta AND
% Autor : Alex Alves

x = [-1 -1 -1 -1 ; 0 0 1 1 ; 0 1 0 1];

% valor desejado 
d= [0 1 1 0;0 0 0 1];

i=1;
maximo =5000;
n=0.8;
a=0.001;
epsilon = 0.000000001;

[NumeroEntradas,NumeroAamostras] = size(x);

% Adicionando o bias a matriz de entrada
x = [-1*ones(1,NumeroAamostras);x];
NumeroEntradas = NumeroEntradas + 1;
% Camada oculta com 4 neuronios mais o bias
pesosCamadaEscondida = rand(5,NumeroEntradas); 
%2 saídas e 5 entradas vindas da camada escondida além do bias
PesosCamadaSaida = rand(2,6);
% Declarando os dentaW
pesosAnterirorCamadaEscondida = zeros(5,NumeroEntradas);
pesosAnteriorCamadaSaida = zeros(2,6);

vetorErro = zeros(1,maximo);
while i < maximo 
    % Muda a ordem das colunas - pra treinar melhor a MLP
    MudaOrdem = randperm(NumeroAamostras);
    x=x(:,MudaOrdem);
    d=d(:,MudaOrdem);
    % Saida da camda escondida
    z = sigmf(pesosCamadaEscondida*x,[1 0]);
    % Adicionando bias
    z= [-1*ones(1,NumeroAamostras);z];
    % Saida da camada de saida 
    y= sigmf(PesosCamadaSaida*z,[1 0]);
    % Calculo do erro
    E = d - y;
    % Erro quadrático médio
    MSE = mean(mean(E.^2));
    vetorErro(i)=MSE;
    if MSE < epsilon
        break;
    else 
        % Ajustes dos pesos da camda de saida 
        [s,N] = size(PesosCamadaSaida);
        %derivadaSigmoide = sigmf(y,[1 0]).*( 1 - sigmf(y,[1 0]));
        derivadaSigmoide = y.*(1-y);
        produto =  derivadaSigmoide.*E;  % produto ponto a ponto
        DeltaWSaida = ((n/N)*produto *z') + (a*pesosAnteriorCamadaSaida);  
        temp = PesosCamadaSaida;
        PesosCamadaSaida = PesosCamadaSaida + DeltaWSaida;  
        pesosAnteriorCamadaSaida= temp;
        % Ajustes dos pesos da camada oculta
        [so,No] = size(pesosCamadaEscondida);
        %derivadaSigmoideCamadaOculta = sigmf(z,[1 0]).*( 1 - sigmf(z,[1 0]));
        derivadaSigmoide = z.*(1-z);
        %derivadaSignoideCamadaOculta =  derivadaSignoideCamadaOculta(2:end,:);
        B =PesosCamadaSaida'*produto;       
        produtoCamdaOculta = derivadaSigmoide.*B; 
        produtoCamdaOculta= produtoCamdaOculta(2:end,:);
        DeltaWCamdaEscondida= ((n/No)*(produtoCamdaOculta *x'))+ (a*pesosAnterirorCamadaEscondida);
        temp = pesosCamadaEscondida;
        pesosCamadaEscondida = pesosCamadaEscondida +  DeltaWCamdaEscondida;  
        pesosAnterirorCamadaEscondida= temp;
    end
    i =i +1;
end    
disp(MSE);
disp(i)
v=1:i;
j=1;

vetY= zeros(1,i);
while j<=i
    vetY(j) = vetorErro(j);
    j=j+1;
end
disp(y)
plot(v, vetY);
disp( pesosCamadaEscondida)
disp(PesosCamadaSaida)