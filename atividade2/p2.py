import numpy as np

print('=== Jacobi ===\n')
def jacobi(A, B):
    # Método de Jacobi (Gladston Moreira)
    """
    input A = nxn array
          B = nx1 array
    """
    # Verifica se o sistema é compatível
    d = np.linalg.det(A)
    if d == 0:
        print('Sistema não é compatível')

    # Estimativa Inicial
    n = len(A)
    X = np.zeros((n, 1))
    Xk = np.copy(X)
    
    # Verifica condições de convergência
    for i in range(n):
        L = 0 # variável que conterá a soma dos elementos "não-pivô" de uma linha
        C = 0 # variável que conterá a soma dos elementos "não-pivô" de uma coluna

        # Somando os elementos que não são pivôs das linhas
        for j in range(n):
            if j != i:
                L += abs(A[i, j])

        # Somando os elementos que não são pivôs das colunas
        for k in range(n):
            if k != i:
                C += abs(A[k, i])

        if L > abs(A[i, i]):
            print('Linha',i+1, ' não satisfaz critério das linhas')
            break
        elif C > abs(A[i, i]):
            print('Coluna',i+1, ' não satisfaz critério das colunas')
            break
          
    # # Critérios de parada
    maxiter = 10
    minDelta = 1e-09
    delta = 1
    
    # # Contador
    k = 0
    print('\nk -\t X \t\t- \t\t max |X(k) - X(k-1)| ')
    print(k,'\t',np.transpose(X),'\t\t\t---')
    while k < maxiter and delta > minDelta:
        # Para cada linha
        for i in range(n):
            # Multiplica e soma elementos conhecidos e joga para o outro lado
            soma = B[i, 0]
            for j in range(n): # Faz isso para cada coluna da linha atual
                if i != j:
                    # Xk irá armazenar o valor das variáveis x na iteração k-1
                    soma += (A[i, j] * (-1)) * Xk[j, 0]
            soma /= A[i, i]
            X[i, 0] = soma
        
        # Calcula max|x(i)k - x(i)k-1|
        lista_auxiliar = [] # Armazena os valores de |x(i)k - x(i)k-1| para cada linha da matriz X
        for i in range(n):
            lista_auxiliar.append(abs(X[i, 0] - Xk[i, 0]))
        delta = max(lista_auxiliar)

        # Iteração
        k = k + 1
        # print k -- xk   --  max|x(i)k-x(i)k-1|
        print(k,'\t',np.transpose(X),'\t\t\t',delta)
        Xk = np.copy(X) # Xk irá armazenar o valor das variáveis x na iteração k-1
    
    print('X:')
    print(X)
    return X

# Teste
print('= Teste 1 =\n')
A = np.array([[1, -1, -1],
              [1, -8, -1],
              [1, -1, -8]])
B = np.array([[2],
              [3],
              [0]])
X = jacobi(A, B)
print('\n')

# Teste 2: Exemplo do slide
print('= Teste 2 =\n')
A = np.array([[1, 2, -2],
              [1, 1, 1],
              [2, 2, 1]])
B = np.array([[1],
              [1],
              [1]])
X = jacobi(A, B)
print('\n')

# Teste 3: Exemplo do slide
print('= Teste 3 =\n')
A = np.array([[0.5, 0.3, 0.6],
              [1, 1, 1],
              [0.4, -0.4, 1]])
B = np.array([[0.2],
              [0],
              [-0.6]])
X = jacobi(A, B)
print('\n')

###########################################################
print('=== Gauss-Seidel ===\n')
def gaussseidel(A,B):
    # Método de GaussSeidel (Gladston Moreira)
    """
    input A = nxn array
          B = nx1 array
    """
    # Verifica se o sistema é compatível
    d = np.linalg.det(A)
    if d==0:
        print('Sistema não é compatível')

    # Estimativa Inicial
    n = len(A)
    X = np.zeros((n, 1))
    Xk = np.copy(X)
    
    # Verifica condições de convergência
    for i in range(n):
        L = 0 # variável que conterá a soma dos elementos "não-pivô" de uma linha
        C = 0 # variável que conterá a soma dos elementos "não-pivô" de uma coluna

        # Somando os elementos das linhas
        for j in range(n):
            if j != i:
                L += abs(A[i, j])

        # Somando os elementos das colunas
        for k in range(n):
            if k != i:
                C += abs(A[k, i])

        if L > abs(A[i, i]):
            print('Linha', i+1, ' não satisfaz critério das linhas')
            break
        elif C > abs(A[i, i]):
            print('Coluna', i+1, ' não satisfaz critério das colunas')
            break
    
    #Critérios de parada
    maxiter = 10
    minDelta = 1e-09
    delta = 1
    
    # Contador
    k = 0
    print('\nk -\t X \t\t- \t\t max |X(k) - X(k-1)| ')
    print(k,'\t',np.transpose(X),'\t\t\t---')
    while k < maxiter and delta > minDelta:
        # Para cada linha
        for i in range(n):
            # Quantidade de componentes atualizados que serão utilizados no cálculo de x(i)k
            numero_componentes_atualizados = i
            # Multiplica e soma elementos conhecidos e joga para o outro lado
            soma = B[i, 0]
            for j in range(n): # Faz isso para cada coluna da linha atual
                if i != j: # Xk irá armazenar o valor das variáveis x na iteração k-1
                    if numero_componentes_atualizados > 0:
                        soma += (A[i, j] * (-1)) * X[j, 0]
                        numero_componentes_atualizados -= 1
                    else:
                        soma += (A[i, j] * (-1)) * Xk[j, 0]
            soma /= A[i, i]
            X[i, 0] = soma
        
        # calcula max|x(i)k-x(i)k-1|
        lista_auxiliar = [] # Armazena os valores |x(i)k - x(i)k-1| para cada linha de X
        for i in range(n):
            lista_auxiliar.append(abs(X[i, 0] - Xk[i, 0]))
        delta = max(lista_auxiliar)

        # interação
        k = k + 1
        # print k -- xk   --  max|x(i)k-x(i)k-1|
        print(k,'\t',np.transpose(X),'\t\t\t',delta)
        Xk = np.copy(X) # Xk irá armazenar o valor das variáveis x na iteração k-1
        
    print('X:')
    print(X)
    return X

# Teste
print('= Teste 1 =\n')
A = np.array([[1, -1, -1],
              [1, -8, -1],
              [1, -1, -8]])
B = np.array([[2],
              [3],
              [0]])
X = gaussseidel(A, B)
print('\n')

# Teste 2: Exemplo do slide
print('= Teste 2 =\n')
A = np.array([[1, -7, 2],
              [8, 1, -1],
              [2, 1, 9]])
B = np.array([[-4],
              [8],
              [12]])
X = gaussseidel(A, B)
print('\n')

# Teste 3: Exemplo do slide
print('= Teste 3 =\n')
A = np.array([[0.5, 0.6, 0.3],
              [1, 1, 1],
              [0.4, -0.4, 1]])
B = np.array([[0.2],
              [0],
              [-0.6]])
X = gaussseidel(A, B)
print('\n')
