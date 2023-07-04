Applicare due volte un filtro equivale ad applicarlo una volta con una dimensione più grande (v.formula dopo). In questo caso, quindi, si ottiene un'immagine molto più sfocata.

$$sigmaUnito=\sqrt{sigma1^2 + sigma2^2}$$


But OpenCV provides scaled rotation with adjustable center of rotation so that you can rotate at any location you prefer. Modified transformation matrix is given by
$$
\begin{bmatrix} \alpha & \beta & (1- \alpha ) \cdot center.x - \beta \cdot center.y + offset_y \\ - \beta & \alpha & \beta \cdot center.x + (1- \alpha ) \cdot center.y +offset_x \end{bmatrix}

where:

\begin{array}{l} \alpha = scale \cdot \cos \theta , \\ \beta = scale \cdot \sin \theta \end{array}
$$
To find this transformation matrix, OpenCV provides a function, cv2.getRotationMatrix2D. Check below example which rotates the image by 90 degree with respect to center without any scaling.


Per cumulare le operazioni $[x',y',1] = (A\cdot B \cdot C)[x,y,1]$


Spiegare e mostrare con un esempio numerico i vantaggi e gli svantaggi nell'applicare un filtro 5x5 oppure due filtri 5x1 e 1x5 in sequenza.

Applicare in cascata due filtri 5x1 e 1x5 in luogo di uno 5x5 ha un vantaggio computazionale. Una convoluzione normale 5x5 richiede O(MNPQ) con M,N dimensione dell'immagine e P,Q dimensione del kernel. Applicare due volte i filtri invece comporta O(MN(P+Q)). Supponiamo una immagine 100x100. Applicare un filtro 5x5 ha un upper bound di operazioni pari a: