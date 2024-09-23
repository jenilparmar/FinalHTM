from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="t5-small")

# Sample paragraph
paragraph = """
We have studied electric field in the last section. It is a vector quantity
and can be represented as we represent vectors. Let us try to represent E
due to a point charge pictorially. Let the point charge be placed at the
origin. Draw vectors pointing along the direction of the
electric field with their lengths proportional to the strength
of the field at each point. Since the magnitude of electric
field at a point decreases inversely as the square of the
distance of that point from the charge, the vector gets
shorter as one goes away from the origin, always pointing
radially outward. Figure 1.12 shows such a picture. In
this figure, each arrow indicates the electric field, i.e., the
force acting on a unit positive charge, placed at the tail of
that arrow. Connect the arrows pointing in one direction
and the resulting figure represents a field line. We thus
get many field lines, all pointing outwards from the point
charge. Have we lost the information about the strength
or magnitude of the field now, because it was contained
in the length of the arrow? No. Now the magnitude of the
field is represented by the density of field lines. E is strong
near the charge, so the density of field lines is more near
the charge and the lines are closer. Away from the charge,
the field gets weaker and the density of field lines is less,
resulting in well-separated lines.
Another person may draw more lines. But the number of lines is not
important. In fact, an infinite number of lines can be drawn in any region.
FIGURE 1.12 Field of a point charge. EXAMPLE 1.8
2024-25 2024-25
20
Physics
It is the relative density of lines in different regions which is
important.
We draw the figure on the plane of paper, i.e., in twodimensions but we live in three-dimensions. So if one wishes
to estimate the density of field lines, one has to consider the
number of lines per unit cross-sectional area, perpendicular
to the lines. Since the electric field decreases as the square of
the distance from a point charge and the area enclosing the
charge increases as the square of the distance, the number
of field lines crossing the enclosing area remains constant,
whatever may be the distance of the area from the charge.
We started by saying that the field lines carry information
about the direction of electric field at different points in space.
Having drawn a certain set of field lines, the relative density
(i.e., closeness) of the field lines at different points indicates
the relative strength of electric field at those points. The field
lines crowd where the field is strong and are spaced apart
where it is weak. Figure 1.13 shows a set of field lines. We
can imagine two equal and small elements of area placed at points R and
S normal to the field lines there. The number of field lines in our picture
cutting the area elements is proportional to the magnitude of field at
these points. The picture shows that the field at R is stronger than at S.
To understand the dependence of the field lines on the area, or rather
the solid angle subtended by an area element, let us try to relate the
area with the solid angle, a generalisation of angle to three dimensions.
Recall how a (plane) angle is defined in two-dimensions. Let a small
transverse line element Dl be placed at a distance r from a point O. Then
the angle subtended by Dl at O can be approximated as Dq = Dl/r.
Likewise, in three-dimensions the solid angle* subtended by a small
perpendicular plane area DS, at a distance r, can be written as
DW = DS/r
2
. We know that in a given solid angle the number of radial
field lines is the same. In Fig. 1.13, for two points P1
 and P2
 at distances
r
1
 and r
2
from the charge, the element of area subtending the solid angle
DW is 2
1
r DW at P1
 and an element of area 2
2
r DW at P2
, respectively. The
number of lines (say n) cutting these area elements are the same. The
number of field lines, cutting unit area element is therefore n/( 2
1
r DW) at
P1
 and n/( 2
2
r DW) at P2
, respectively. Since n and DW are common, the
strength of the field clearly has a 1/r
2
 dependence.
The picture of field lines was invented by Faraday to develop an
intuitive non-mathematical way of visualising electric fields around
charged configurations. Faraday called them lines of force. This term is
somewhat misleading, especially in case of magnetic fields. The more
appropriate term is field lines (electric or magnetic) that we have
adopted in this book.
Electric field lines are thus a way of pictorially mapping the electric
field around a configuration of charges. An electric field line is, in general,
"""

# Generate the summary
summary = summarizer(paragraph, max_length=1500, min_length=200, do_sample=False)

# Print the summary
print("Summary:")
print(len(summary[0]['summary_text']))
print((summary[0]['summary_text']))
