.. _permutation-algorithms:

Permutation Algorithms
======================

When computing symmetry overlaps for molecular geometries, PoSym must
determine how atoms permute under each symmetry operation. This is the
permutation assignment problem.

The Problem
-----------

Given a set of atoms with coordinates :math:`\{\mathbf{r}_i\}` and
symbols :math:`\{s_i\}`, and a symmetry operation :math:`\hat{R}`
(rotation, reflection, etc.), we need to find the permutation :math:`\pi`
that maps each atom to its image under :math:`\hat{R}`:

.. math::

   \langle \mathbf{P} | \hat{R} \mathbf{P} \rangle =
   \sum_i \langle \mathbf{r}_i | \hat{R} \mathbf{r}_{\pi(i)} \rangle

The permutation must be consistent with the group multiplication table:
the permutation for each operation must satisfy the same relations as
the corresponding 3x3 matrices.

Hungarian Algorithm (Default)
-----------------------------

The default method uses the Hungarian algorithm (linear sum assignment)
via :func:`~posym.permutation.hungarian.get_permutation_hungarian`:

.. code-block:: python

   Configuration().algorithm = 'hungarian'

This is fast but approximate. For each generator operation, the algorithm:

1. Computes the cost matrix :math:`C_{ij} = -\langle \mathbf{r}_i | \hat{R} \mathbf{r}_j \rangle`
2. Solves the assignment problem to minimize :math:`\sum_i C_{i, \pi(i)}`
3. Restricts permutations to atoms with the same label

The Hungarian algorithm is :math:`\mathcal{O}(n^3)` in the number of
atoms, making it suitable for large systems.

Exact Algorithm
---------------

For cases where the Hungarian approximation is insufficient:

.. code-block:: python

   Configuration().algorithm = 'exact'

The exact method enumerates all valid permutation assignments for the
group generators that satisfy:

- Atom labels must match after permutation
- Orbit structures from generator permutations must be compatible
- The composed permutations (corresponding to non-generator operations)
  must respect the group's multiplication table

For each valid permutation set, the full symmetry measure is computed
and the set maximizing the totally symmetric character is selected.

The exact algorithm is :math:`\mathcal{O}(n! / m!)` in the worst case
and should be used only for small systems or when high accuracy is required.

Label Tolerance
---------------

The :attr:`~posym.config.Configuration.label_tolerance` parameter
controls how chemically equivalent atoms are distinguished based on
their distance from the molecular center:

.. code-block:: python

   Configuration().label_tolerance = 0.1  # Angstrom

Atoms with eigenvector degeneracies below this tolerance are assigned
the same label, reducing the permutation search space.
