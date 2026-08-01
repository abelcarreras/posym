Continuous Symmetry Measures
=============================

PoSym implements the continuous symmetry measure (CSM) formalism
introduced by Pinsky, Dryzun, Casanova, Alemany, and Avnir
(:cite:t:`Pinsky2008`, :cite:t:`Pinsky2008b`).

Definition
----------

The CSM quantifies how far a given object is from perfect symmetry
with respect to a point group :math:`G`. For a molecular geometry
represented by atomic coordinates :math:`\mathbf{P} = \{\mathbf{r}_i\}`,
the CSM is defined as:

.. math::

   S(G) = 100 \times \min_{\mathbf{P}_G}
          \frac{\|\mathbf{P} - \mathbf{P}_G\|^2}{\|\mathbf{P}\|^2}

where :math:`\mathbf{P}_G` is the nearest perfectly symmetric
structure. This minimization is equivalent to averaging the structure
over all group operations after finding the optimal orientation.

Operational Form
----------------

In the operational form used by PoSym, the CSM is computed from the
overlaps between the original structure and its images under group
operations:

.. math::

   S(G) = 100 \times \left(1 - \frac{1}{N_G}
          \frac{\sum_{g \in G} \langle \mathbf{P} | g\mathbf{P} \rangle}
               {\langle \mathbf{P} | \mathbf{P} \rangle} \right)

where :math:`N_G` is the group order, and :math:`\langle \mathbf{P} | g\mathbf{P} \rangle`
is the dot product of atomic positions.

Irreducible Representation Decomposition
----------------------------------------

The overlaps vector defined in symmetry operations basis are transformed into the
irreducible representation basis of the point group. This gives a vector in the G-module space:

.. math::

   \chi(G) = M^{-1} \cdot \vec{s}

where :math:`\vec{s}` is the vector of operation overlaps (one entry
per class), and :math:`M^{-1}` is the inverse of the character table
matrix.

This decomposition reveals the distribution of symmetry content:
the coefficient of the totally symmetric representation (e.g.,
:math:`A_1`, :math:`A_{1g}`) is the measure of how close the structure
is to perfect symmetry.

Beyond Geometry
---------------

The same formalism is applied to other objects (wave functions, densities)
by replacing the dot product with a general definition of the inner product
for the corresponding object type:

.. math::

   s(g) = \frac{\langle \psi | \hat{g} \psi \rangle}
               {\langle \psi | \psi \rangle}

This allows PoSym to analyze the symmetry of molecular orbitals,
densities, Slater determinants, wave functions, etc.
