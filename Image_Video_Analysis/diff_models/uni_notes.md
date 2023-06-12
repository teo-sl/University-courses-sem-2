# Introduzione

A differenza dei VAE, dove la trasformazione da rumore a immagine avviene in un unico passo, qui consideriamo dei passi successivi di denoising modellati atteaverso delle markov chain.

Servono due processi
- forward: deterministica, e consiste nello sporcare l'immagine
- backward: intrattabile, approssimata attraverso una rete neurale (Unet)


$$q(x_t|x_{t-1})=\prod_{t=1}^T q(x_t|x_{t-1}), \;\;\; \{\beta_t \in (0,1)\}_{t=1}^T$$

beta governa il rumore, indica il rumore da iniettare a ogni passo. Solitamente è precalcolato, non vi sono reti per approssimarlo.

Quando $T\rightarrow \infty$ si ottiene una gaussiana ipertrofica, i.e. aggiungiamo talmente tanto rumore da ricondurci alla gaussiana.

$$q(x_t|x_{t-1}) = N(\sqrt{1-\bet
_t}x_{t-1},\beta_t I)$$

Si può direttamente calcolare x al passo t usando una forma chiusa.

$$x_t = \sqrt{1-\bet
_t}x_{t-1}+\sqrt{\beta_t}\epsilon_{t-1} \;\;\; \epsilon \tilde{ N(0,1)}$$

Da cui deriva:

$$
\sqrt{\hat
{\alpha
_t}}x_0 + \sqrt{1-\hat
{\alpha_t}\epsilon}
$$

Dove

$$
\alpha_t = 1-\beta_t \;\;\;\hat{\alpha_t} = \prod_{i=1}^t \alpha_i 
$$


# Reverse

Idealmente vorremmo partire dal rumore gaussiano e tornare indietro all'immagine originale usando ancora $q$. Questo problema è intrattabile, lo approssimiamo con una rete neurale ($p_\theta$).

Quello che facciamo è il denoising.


Trovare l'esatta distribuzione è difficile. Per Bayes:

$$
q(x_{t-1}|x_t) = q(x_t | x_{t-1} \frac{q(x_{t-1})}{q(x_t)})
$$

$$

q(x_t) = \int_{-\infty}^\infty q(x_t|x_{t-1})q(x_)

$$