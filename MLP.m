
% Implementação de MLP com camadas variáveis

% Autor : Alex Alves

function saida = Main()
 
   x= -4*pi:0.1:4*pi;
   for c=1:length(x)
        d(c) = sin(x(c))/x(c);
   end

   figure
   plot(d);
    i=1;
    maximo =5000; % Número máximo de iterações
    n=0.8;  % taxa de treinamento
    a=0.001; % Momento
    epsilon = 0.000000001; % Valor de erro aceitável
    entradas = 8;
    saidas =1;
    camadas = 4;
    %passar pesos 
    [VetorMSE,pesosCamadaEscondida,PesosCamadaSaida] = Treinar(x,d,n,a,entradas, maximo, epsilon,i,saidas,camadas);	
    % Executa a MLP
    saida= Executar(x,pesosCamadaEscondida,PesosCamadaSaida,camadas);
end
function [vetY,pesosCamadaEscondida,PesosCamadaSaida]= Treinar(x,d,n,a,entradas, maximo, epsilon,i,saidas,camadas)

    [NumeroEntradas,NumeroAamostras] = size(x);

    % Adicionando o bias a matriz de entrada
    x = [-1*ones(1,NumeroAamostras);x];
    NumeroEntradas = NumeroEntradas + 1;
    % Camada oculta com 4 neuronios mais o bias
    pesosCamadaEscondida = rand(entradas,NumeroEntradas,camadas);  % Mexe aki
    %2 saídas e 5 entradas vindas da camada escondida além do bias
    PesosCamadaSaida = rand(saidas,entradas+1);
    % Declarando os deltaW
    pesosAnterirorCamadaEscondida = zeros(entradas,NumeroEntradas,camadas); % Mexe aki
    pesosAnteriorCamadaSaida = zeros(saidas,entradas+1);

    vetorErro = zeros(1,maximo);

    while i < maximo 
        % Muda a ordem das colunas - pra treinar melhor a MLP
        MudaOrdem = randperm(NumeroAamostras);
        x=x(:,MudaOrdem);
       % d=d(:,MudaOrdem);
       
        d=d(1,MudaOrdem);
        % Saida da camda escondida
        z = sigmf(pesosCamadaEscondida(:,:,camadas)*x,[1 0]);
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
            derivadaSigmoide = y.*(1-y);
            produto =  derivadaSigmoide.*E;  % produto ponto a ponto
            DeltaWSaida = ((n/N)*produto *z') + (a*pesosAnteriorCamadaSaida);  
            temp = PesosCamadaSaida;
            PesosCamadaSaida = PesosCamadaSaida + DeltaWSaida;  
            pesosAnteriorCamadaSaida= temp;
            for u=1:camadas
                % Ajustes dos pesos da camada oculta
                % aqui fica o for das camadas escondidas
                % camada de saida usa a mesma 
                [so,No,s1] = size(pesosCamadaEscondida);
                derivadaSigmoide = z.*(1-z);

                B =PesosCamadaSaida'*produto;  
                
                produtoCamdaOculta = derivadaSigmoide.*B; 
                produtoCamdaOculta= produtoCamdaOculta(2:end,:);
                DeltaWCamdaEscondida= ((n/No)*(produtoCamdaOculta *x'))+ (a*pesosAnterirorCamadaEscondida(:,:,u));
                         
                temp = pesosCamadaEscondida(:,:,u);
                pesosCamadaEscondida(:,:,u) = pesosCamadaEscondida(:,:,u) +  DeltaWCamdaEscondida;  
                pesosAnterirorCamadaEscondida(:,:,u)= temp;
            end
        end
        i =i +1;
    end    
    v=1:i;
    j=1;

    vetY= zeros(1,i);
    while j<=i
        vetY(j) = vetorErro(j);
        j=j+1;
    end
    figure
    plot(v, vetY);
 

end
function  y= Executar(x,pesosCamadaEscondida,PesosCamadaSaida,camadas)

   [NumeroEntradas,NumeroAamostras] = size(x);
    x = [-1*ones(1,NumeroAamostras);x];
    % Saida da camda escondida

    z = sigmf(pesosCamadaEscondida(:,:,camadas)*x,[1 0]);
    % Adicionando bias
    z= [-1*ones(1,NumeroAamostras);z];
    % Saida da camada de saida 
    y= sigmf(PesosCamadaSaida*z,[1 0]);
    
    figure
    plot(y);
end

