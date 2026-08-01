Irreducible Representation Decomposition
========================================

The core mathematical operation in PoSym is the decomposition of an
arbitrary vector of symmetry operation characters into the irreducible
representation (IR) basis of a point group.

Character Tables
----------------

For a point group :math:`G` with irreducible representations
:math:`\Gamma_1, \Gamma_2, \ldots, \Gamma_r` and operation classes
:math:`C_1, C_2, \ldots, C_k`, the character table is a :math:`r \times k`
matrix :math:`M`:

.. math::

   M_{ij} = \chi^{\Gamma_i}(C_j)

where :math:`\chi^{\Gamma_i}(C_j)` is the character of irreducible
representation :math:`\Gamma_i` under operation class :math:`C_j`.

.. _transform-matrices:

Transformation Between Bases
----------------------------

PoSym uses two reference bases:

**Operator basis**: A vector of operation overlaps :math:`\vec{s} = (s_1, s_2, \ldots, s_k)`
where :math:`s_j` is the overlap under operation class :math:`C_j`.

**IR basis**: A vector of IR coefficients :math:`\vec{c} = (c_1, c_2, \ldots, c_r)`
where :math:`c_i` is the contribution of :math:`\Gamma_i`.

The transformation is:

.. math::

   \vec{s} = M \cdot \vec{c} \quad \text{(IR → Op)}

.. math::

   \vec{c} = M^{-1} \cdot \vec{s} \quad \text{(Op → IR)}

