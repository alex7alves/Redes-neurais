
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
    maximo =25000; % Número máximo de iterações
    n=0.8;  % taxa de treinamento
    a=0.001; % Momento
    epsilon = 0.000000001; % Valor de erro aceitável
    entradas = 8;
    saidas =1;
    camadas = 4;  % numero cadas escondida
    %passar pesos 
    [VetorMSE,pesosCamadaEntrada,pesosCamadaEscondida,PesosCamadaSaida] = Treinar(x,d,n,a,entradas, maximo, epsilon,i,saidas,camadas);	
    % Executa a MLP
    saida= Executar(x,pesosCamadaEntrada,pesosCamadaEscondida,PesosCamadaSaida,camadas);
end
function [vetY,pesosCamadaEntrada,pesosCamadaEscondida,PesosCamadaSaida]= Treinar(x,d,n,a,neuronios, maximo, epsilon,i,saidas,camadas)

    [NumeroEntradas,NumeroAamostras] = size(x);

    % Adicionando o bias a matriz de entrada
    x = [-1*ones(1,NumeroAamostras);x];
    NumeroEntradas = NumeroEntradas + 1;
    % Camada de entrada
    pesosCamadaEntrada = rand(neuronios,NumeroEntradas); 
    % Camadas escondidas 
    pesosCamadaEscondida = rand(neuronios,neuronios+1,camadas);  
    %2 saídas e 5 entradas vindas da camada escondida além do bias
    PesosCamadaSaida = rand(saidas,neuronios+1);
    % Declarando os deltaW
    pesosAnterirorCamadaEntrada =zeros(neuronios,NumeroEntradas);
    pesosAnterirorCamadaEscondida = zeros(neuronios,neuronios+1,camadas);
    pesosAnteriorCamadaSaida = zeros(saidas,neuronios+1);
    
    zlayer= zeros(neuronios+1,NumeroAamostras,camadas);
    vetorErro = zeros(1,maximo);

    while i < maximo 
        % Muda a ordem das colunas - pra treinar melhor a MLP
        MudaOrdem = randperm(NumeroAamostras);
        x=x(:,MudaOrdem);
        % d=d(:,MudaOrdem);
        d=d(1,MudaOrdem);
        z = sigmf(pesosCamadaEntrada*x,[1 0]);
        zCamadaEscondida = [-1*ones(1,NumeroAamostras);z];
        %forward/propagation
        for u=1:camadas          
                 zCamadaEscondida = sigmf(pesosCamadaEscondida(:,:,u)*zCamadaEscondida,[1 0]);
                 zCamadaEscondida = [-1*ones(1,NumeroAamostras); zCamadaEscondida];
                 zlayer(:,:,u)= zCamadaEscondida;
        end

        % Saida da camada de saida 
        y= sigmf(PesosCamadaSaida*zCamadaEscondida,[1 0]); %linear
        % Calculo do erro
        E = d - y;
        % Erro quadrático médio
        MSE = mean(mean(E.^2));
        vetorErro(i)=MSE;
        if MSE < epsilon
            break;
        else 
            %backpropagation
            % Ajustes dos pesos da camda de saida 
            [s,N] = size(PesosCamadaSaida);
            derivadaSigmoide = y.*(1-y);
            produto =  derivadaSigmoide.*E;  % produto ponto a ponto   1x252
            DeltaWSaida = ((n/N)*produto *zlayer(:,:,camadas)') + (a*pesosAnteriorCamadaSaida);  %1x9
            temp = PesosCamadaSaida;
            PesosCamadaSaida = PesosCamadaSaida + DeltaWSaida;  
            pesosAnteriorCamadaSaida= temp;   %1x9
            
            %Backpropagation das camadas internas

            B =PesosCamadaSaida'*produto;   %9x1 vs 1x252 = 9vs252 
            
            [so,No,s1] = size(pesosCamadaEscondida);
            for u=camadas:-1:1
                
                derivadasigmoideCamadasEntradas= zlayer(:,:,u).*(1-zlayer(:,:,u));
                if u==camadas
                    produtoCamdaOculta = derivadasigmoideCamadasEntradas.*B;
                else
                    produtoCamdaOculta = derivadasigmoideCamadasEntradas.*produtoCamdaOculta;
                end
                 
                produtoCamdas= produtoCamdaOculta(2:end,:);
              
                if u>1
                    DeltaWCamdaEscondida= ((n/No)*(produtoCamdas *zlayer(:,:,u-1)'))+ (a*pesosAnterirorCamadaEscondida(:,:,u));
                    temp = pesosCamadaEscondida(:,:,u);
                    pesosCamadaEscondida(:,:,u) = pesosCamadaEscondida(:,:,u) +  DeltaWCamdaEscondida;  
                    pesosAnterirorCamadaEscondida(:,:,u)= temp;
                else
                    DeltaWCamdaEscondida= ((n/No)*(produtoCamdas *x'))+ (a* pesosAnterirorCamadaEntrada);
                    temp = pesosCamadaEntrada;
                    pesosCamadaEntrada = pesosCamadaEntrada +  DeltaWCamdaEscondida;  
                    pesosAnterirorCamadaEntrada= temp;

                end
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
function  y= Executar(x,pesosCamadaEntrada,pesosCamadaEscondida,PesosCamadaSaida,camadas)

   [NumeroEntradas,NumeroAamostras] = size(x);
    x = [-1*ones(1,NumeroAamostras);x];
    % Saida da camda escondida
    
    z = sigmf(pesosCamadaEntrada*x,[1 0]);
    zCamadaEscondida = [-1*ones(1,NumeroAamostras);z];

    for u=1:camadas           
        zCamadaEscondida = sigmf(pesosCamadaEscondida(:,:,u)*zCamadaEscondida,[1 0]);
        zCamadaEscondida = [-1*ones(1,NumeroAamostras); zCamadaEscondida];
        zlayer(:,:,u)= zCamadaEscondida;
    end
        
     % Saida da camada de saida 
     y= sigmf(PesosCamadaSaida*zCamadaEscondida,[1 0]); %linear
    
    figure
    plot(y);
end

