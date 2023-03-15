# 1. Introduzione

Abbiamo un grafo che ci dice quale sarà la transizione di un agente step dopo step. Ogni arco ha una probabilità. Un problema consiste nel vedere verso cosa convergerà il comportamento degli agenti.

Rappresentiamo il grafo come una matrice di adiacenza. Nella posizione (i,j) è presente la probabilità che nel prossimo step l'agente passi dal nodo i al nodo j. Associamo, per ogni nodo i definiamo il numero di agenti che si trovano in quel nodo allo step 0

$$A \cdot T_0 $$

Dove $T_0$ è un vettore colonna con i valori iniziali. Il risultato del prodotto è un vettore colonna che corrisponde a $T_1$.

Quindi, in generale vale:

$$T_{i+1} = A \cdot T_i$$

Naturalmente, il numero di agenti nel vettore $T_i$ è espresso in percentuale, come valore $\in [0,1]$. Il prossimo stato è detto next state o future state.

# 2. Markov Chain

Perché sono chiamate chains? Si basa sulle successive moltiplicazioni
Ogni volta, il vettore risultante deve essere tale che la somma degli elementi sia 1.

# 3. Metodo alternativo

Un modo alternativo è fare così:

$$T_{i} = A^{i} \cdot T_0$$


Un aspetto interessante è che il vettore sembra variare molto velocemente all'inizio per poi convergere a un vettore stabile finale.

Se la matrice cambia, il vettore dovrà convergere nuovamente a un nuovo valore. 

# 4. Stocasticità e regolarità

## 4.1 Stocasticità

Una matrice è stocastica se la somma delle righe/colonne è 1. Per usare le markov chain le matrici devono essere stocastiche. 

Se è 1 se si sommano le colonne

$$B = P \cdot A$$

Se è 1 se si sommano le righe

$$B = A \cdot P$$

Con A vettore colonna e P matrice di transizione.


## 4.2 Regolarità

Una matrice stocastica è anche regolare se $P^n$ $(n>1)$ ha solo entry > 0, quindi non negative e diverse da zero. 


# 5. Markov chain regolari
Se P è una matrice regolare ci sarà un $P^n$ dove $P^n X_0 = \hat{X}$ dove $\hat{X}$ è la stable distribution matrix. O dove $P^{n+1} = P^n$, allora $P,P^2,P^3,...$ è una markov chain regolare.

Quindi, nel caso delle markov chain regolari, per ottenere il vettore a convergenza basta moltiplicare lo stato iniziale con la matrice di transizione a convergenza (la stable distribuzion matrix).











