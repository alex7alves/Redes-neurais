
% Implementação de MLP com camadas variáveis

% Autor : Alex Alves

% fiz funçao falta colocr pro treinar
% falta fazer funcao derivada


function saida = Main()
 
   x= -4*pi:0.1:4*pi;
   for c=1:length(x)
        d(c) = sin(x(c))/x(c);
   end
  
%x = [0 0 1 1 ; 0 1 0 1];  %n=0.8, a=0.001, entradas=4,camadas=2
%d= [0 1 1 0;0 0 0 1];
   figure
   plot(d);
   disp(d(1,239:252));
    i=1;
    maximo =10000; % Número máximo de iterações
    n=0.5;  % taxa de treinamento
    a=0.001; % Momento
    epsilon = 0.000000001; % Valor de erro aceitável
    entradas = 4;
    saidas =1;
    camadas = 2;  % numero cadas escondida
    p=0.041;  % pra função atan
    tipo=3;
    %passar pesos 
    [VetorMSE,pesosCamadaEntrada,pesosCamadaEscondida,PesosCamadaSaida] = Treinar(x,d,n,a,entradas, maximo, epsilon,i,saidas,camadas,p,tipo);	
    % Executa a MLP
    saida= Executar(x,pesosCamadaEntrada,pesosCamadaEscondida,PesosCamadaSaida,camadas,p,tipo);
end
function [vetY,pesosCamadaEntrada,pesosCamadaEscondida,PesosCamadaSaida]= Treinar(x,d,n,a,neuronios, maximo, epsilon,i,saidas,camadas,p,tipo)

    [NumeroEntradas,NumeroAamostras] = size(x);

    % Adicionando o bias a matriz de entrada
    x = [-1*ones(1,NumeroAamostras);x];
    NumeroEntradas = NumeroEntradas + 1;
    % Camada de entrada
    pesosCamadaEntrada = rand(neuronios,NumeroEntradas); 
    % Camadas escondidas 
    pesosCamadaEscondida = rand(neuronios,neuronios+1,camadas);  
    %Camada de saida
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
        z=funcao(p,pesosCamadaEntrada,x,tipo);
        zCamadaEscondida = [-1*ones(1,NumeroAamostras);z];
        %forward/propagation
        for u=1:camadas  
            zCamadaEscondida = funcao(p,pesosCamadaEscondida(:,:,u),zCamadaEscondida,tipo);
            zCamadaEscondida = [-1*ones(1,NumeroAamostras); zCamadaEscondida];
            zlayer(:,:,u)= zCamadaEscondida;
        end

        % Saida da camada de saida 
        y= funcao(p,PesosCamadaSaida,zCamadaEscondida,tipo); 
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
           % derivadaSigmoide = y.*(1-y);
           derivadaSigmoide = p./(1+(y.^2));
            produto =  derivadaSigmoide.*E;  % produto ponto a ponto   1x252
           % DeltaWSaida = ((n/N)*produto *zlayer(:,:,camadas)') + (a*pesosAnteriorCamadaSaida);  %1x9
            DeltaWSaida = ((n)*produto *zlayer(:,:,camadas)') + (a*pesosAnteriorCamadaSaida);  %1x9
            temp = PesosCamadaSaida;
            PesosCamadaSaida = PesosCamadaSaida + DeltaWSaida;  
            pesosAnteriorCamadaSaida= temp;   %1x9
            
            %Backpropagation das camadas internas

            B =PesosCamadaSaida'*produto;   %9x1 vs 1x252 = 9vs252 
            
            [so,No,s1] = size(pesosCamadaEscondida);
            for u=camadas:-1:1
                
               %  derivadasigmoide= zlayer(:,:,u).*(1-zlayer(:,:,u));
                derivadasigmoide= p./(1+(zlayer(:,:,u)).^2);
                if u==camadas
                    produtoCamdaOculta = derivadasigmoide.*B;
                else
                    produtoCamdaOculta = derivadasigmoide.*produtoCamdaOculta;
                end
                 
                produtoCamdas= produtoCamdaOculta(2:end,:);
              
                if u>1
                  %  DeltaWCamdaEscondida= ((n/No)*(produtoCamdas *zlayer(:,:,u-1)'))+ (a*pesosAnterirorCamadaEscondida(:,:,u));
                    DeltaWCamdaEscondida= ((n)*(produtoCamdas *zlayer(:,:,u-1)'))+ (a*pesosAnterirorCamadaEscondida(:,:,u));
                    temp = pesosCamadaEscondida(:,:,u);
                    pesosCamadaEscondida(:,:,u) = pesosCamadaEscondida(:,:,u) +  DeltaWCamdaEscondida;  
                    pesosAnterirorCamadaEscondida(:,:,u)= temp;
                else
                   % DeltaWCamdaEntrada= ((n/No)*(produtoCamdas *x'))+ (a* pesosAnterirorCamadaEntrada);
                    DeltaWCamdaEntrada= ((n)*(produtoCamdas *x'))+ (a* pesosAnterirorCamadaEntrada);
                    temp = pesosCamadaEntrada;
                    pesosCamadaEntrada = pesosCamadaEntrada +  DeltaWCamdaEntrada;  
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
function  y= Executar(x,pesosCamadaEntrada,pesosCamadaEscondida,PesosCamadaSaida,camadas,p,tipo)

   [NumeroEntradas,NumeroAamostras] = size(x);
    x = [-1*ones(1,NumeroAamostras);x];
    % Saida da camda escondida
    z= funcao(p,pesosCamadaEntrada,x,tipo);
    zCamadaEscondida = [-1*ones(1,NumeroAamostras);z];

    for u=1:camadas           
        zCamadaEscondida=funcao(p,pesosCamadaEscondida(:,:,u),zCamadaEscondida,tipo);
        zCamadaEscondida = [-1*ones(1,NumeroAamostras); zCamadaEscondida];
    end
        
    % Saida da camada de saida 
    y= funcao(p,PesosCamadaSaida,zCamadaEscondid,tipo);   
    figure
    plot(y);
end

function w= funcao(p,wp,f,tipo)

    if tipo==1
        w= sigmf(wp*f,[1 0]);
        
    elseif tipo==2
        w = atan(p*wp*f);
    elseif tipo==3    
        w=tanh(p*wp*f);
   else 
       w= sigmf(wp*f,[1 -1]);
   end
end
