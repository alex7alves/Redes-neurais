
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
    camadas = 4;  % numero cadas escondida
    %passar pesos 
    [VetorMSE,pesosCamadaEscondida,PesosCamadaSaida] = Treinar(x,d,n,a,entradas, maximo, epsilon,i,saidas,camadas);	
    % Executa a MLP
    saida= Executar(x,pesosCamadaEscondida,PesosCamadaSaida,camadas);
end
function [vetY,pesosCamadaEscondida,PesosCamadaSaida]= Treinar(x,d,n,a,neuronios, maximo, epsilon,i,saidas,camadas)

    [NumeroEntradas,NumeroAamostras] = size(x);

    % Adicionando o bias a matriz de entrada
    x = [-1*ones(1,NumeroAamostras);x];
    NumeroEntradas = NumeroEntradas + 1;
    % Camada oculta com 4 neuronios mais o bias
    pesosCamadaEntrada = rand(neuronios,NumeroEntradas); 
    % Camadas escondidas depois da 
    pesosCamadaEscondida = rand(neuronios,neuronios+1,camadas);  % da entrada até 
    %2 saídas e 5 entradas vindas da camada escondida além do bias
    PesosCamadaSaida = rand(saidas,neuronios+1);
    % Declarando os deltaW
    pesosAnterirorCamadaEntrada =zeros(neuronios,NumeroEntradas);
    pesosAnterirorCamadaEscondida = zeros(neuronios,neuronios,camadas); % Mexe aki
    pesosAnteriorCamadaSaida = zeros(saidas,neuronios+1);
    
    zlayer= zeros(saidas,NumeroAamostras,camadas-2); % so as internas sem saida nem entrada
    vetorErro = zeros(1,maximo);

    while i < maximo 
        % Muda a ordem das colunas - pra treinar melhor a MLP
        MudaOrdem = randperm(NumeroAamostras);
        x=x(:,MudaOrdem);
       % d=d(:,MudaOrdem);
       
        d=d(1,MudaOrdem);
        z = sigmf(pesosCamadaEntrada*x,[1 0]);
        zCamadaEscondida = [-1*ones(1,NumeroAamostras);z];
        disp(size(z));
        %forward/propagation
        for u=1:camadas          
                 zCamadaEscondida = sigmf(pesosCamadaEscondida(:,:,u)*zCamadaEscondida,[1 0]);
                 zCamadaEscondida = [-1*ones(1,NumeroAamostras); zCamadaEscondida];
        end
        
        % Adicionando bias
     %   z= [-1*ones(1,NumeroAamostras);z];
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
            DeltaWSaida = ((n/N)*produto *z') + (a*pesosAnteriorCamadaSaida);  %1x9
            temp = PesosCamadaSaida;
            PesosCamadaSaida = PesosCamadaSaida + DeltaWSaida;  
            pesosAnteriorCamadaSaida= temp;   %1x9
            
            %Backpropagation das camadas internas
           
            z = PesosCamadaSaida'*derivadaSigmoide; %9x1 1x252
           % z=aux(2:end,:);
            derivadaSigmoide = z.*(1-z);   %9x252
            disp(size(derivadaSigmoide))
             %B =PesosCamadaSaida'*produto;
            % B =PesosCamadaSaida*produto;
            B =PesosCamadaSaida'*produto;
            
           % produtoCamdaOculta = derivadaSigmoide.*B; 
           produtoCamdaOculta = derivadaSigmoide.*B; 
            produtoCamdaOculta= produtoCamdaOculta(2:end,:);
                %Corrigir ai pegar o z ou peso da entrada(na camada 1)  da
                %entrada anterior. To na 4 pega o z da 3
              
                DeltaWCamdaEscondida= ((n/No)*(produtoCamdaOculta *x'))+ (a*pesosAnterirorCamadaEscondida(:,:,u));
                         
                temp = pesosCamadaEscondida(:,:,camadas-1);
                pesosCamadaEscondida(:,:,camadas-1) = pesosCamadaEscondida(:,:,camadas-1) +  DeltaWCamdaEscondida;  
                pesosAnterirorCamadaEscondida(:,:,camadas-1)= temp;
           %  B=B(2:end,:);
            % disp(size(B))
            for u=camadas-1:-1:1
                if u<(camadas-1)
                    if mod(u,2)==0
                        % DUVIDA 
                        disp(size(derivadaSigmoide))
                         z=pesosCamadaEscondida(:,:,u)'*derivadaSigmoide;
                         derivadaSigmoide = z.*(1-z);
                       %  derivadaSigmoide  = sigmf(pesosCamadaEscondida(:,:,u)'*derivadaSigmoide ,[1 0]);
                    else
                        disp('passou aki')
                         z=pesosCamadaEscondida(:,:,u)*derivadaSigmoide;
                         derivadaSigmoide = z.*(1-z);
                       % derivadaSigmoide  = sigmf(pesosCamadaEscondida(:,:,u)*derivadaSigmoide ,[1 0]);
                    end
                     % DUVIDA 
                     B = pesosCamadaEscondida*z;
                else
                    
                   % B =PesosCamadaSaida'*produto;
                   
                end

                [so,No,s1] = size(pesosCamadaEscondida);
            
                
                produtoCamdaOculta = derivadaSigmoide.*B; 
                produtoCamdaOculta= produtoCamdaOculta(2:end,:);
                %Corrigir ai pegar o z ou peso da entrada(na camada 1)  da
                %entrada anterior. To na 4 pega o z da 3
              
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

    z = sigmf(pesosCamadaEscondida(:,:,camadas-1)*x,[1 0]);
    % Adicionando bias
    z= [-1*ones(1,NumeroAamostras);z];
    % Saida da camada de saida 
    y= sigmf(PesosCamadaSaida*z,[1 0]);
    
    figure
    plot(y);
end

