# Unsupervised-classification
This is the code base of Eivind Salvesens specialization project in unsupervised learning. The goal of the project was to find a suitable unsupervised algorithm for plankton classification. 
\ref{table:results}. 
\begin{table}[ht]
\centering
\caption{Classification results}
 \begin{tabular}{||c c c c||} 
 \hline
  Model & SC & K-means & DCEC \\ [0.5ex] 
 \hline
 Baseline model & 0.61 & 0.44 & -\\ [1ex] 
 \hline
 FullyConnected & 0.52 & 0.49 & 0.52\\ [1ex] 
 \hline
 VGG16-dense& 0.67 & 0.60 & 0.71\\ [1ex] 
 \hline
 VGG16-GAP & 0.73 & 0.55 & 0.62\\ [1ex] 
 \hline
 VGG16-Sparse & 0.70 & 0.50 & 0.70\\ [1ex] 
 \hline
 COAPNET-dense& 0.74 & 0.65 & 0.71\\ [1ex] 
 \hline
 COAPNET-GAP & \textbf{0.76} & 0.62  & \textbf{0.70} \\ [1ex] 
 \hline
 COAPNET-Sparse & 0.71 & 0.53  & 0.67 \\ [1ex] 
 \hline
\end{tabular}
\label{table:results}
\end{table}%
