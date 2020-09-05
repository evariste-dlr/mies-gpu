
# MIES sur GPU

__French below__

CUDA implementation of a multi-scale method for graph reduction, based on the computing of an
independent edge set on a given graph (see [Independent set](https://en.wikipedia.org/wiki/Independent_set_(graph_theory)).

Files and functions are prefixed with `h_` or `d_` depending on if this is a host or device
implementation.

The input graph is given as a sparse adjacency matrix encoded both on the CSR and COO format.
This particular coding is defined in `sparse.[c|h]`. To apply the method to multiple graphs, you
cad define a large adjacency bloc-diagonal matrix in which each block is a particular graph.

`bin/benchmark_mies` generates a set of graphes (with in total 2 000 000 vertices) and apply
the process on 10 graphs at a time to evaluate a mean execution time.

----

Implémentation CUDA d'une méthode multi-échelle de réduction de graphes basée sur le calcul
d'un Maximal Independant Edge Set sur un graphe (voir [Independent set](https://en.wikipedia.org/wiki/Independent_set_(graph_theory)).

Les fichiers ainsi que les fonctions peuvent être préfixées de `h_` ou `d_` suivant si il
s'agit respectivement d'une implémentation sur CPU (host) ou GPU (device).

Le codage du graphe d'entrée est une matrice d'adjacence creuse, à la fois au format CSR et COO.
Ce codage est défini dans `sparse.[cpp|h]`. Pour appliquer la méthode à plusieurs graphes il suffit
de coder ces graphes dans une matrice diagonale par blocs dans laquelle chaque bloc représente
un graphe.

L'exécutable `bin/benchmark_mies` génère un ensemble de graphes (au total 2 000 000 de noeuds) et
applique le traitement sur ces graphes plusieurs fois (10) pour calculer une moyenne de temps.
Les temps moyens CPU et GPU sont donnés en fin d'exécution.

Exemple de sortie :

```
 Generate graphs...
    2000000 nodes
    12400000 edges

 Compute MIES on CPU...
    [MIES]
   0.37017 s

 Compute MIES on GPU...
    [MIES]
   0.0690053 s
```
