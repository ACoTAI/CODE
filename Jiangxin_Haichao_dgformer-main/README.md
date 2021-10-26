DGformer
============================================



A **PyTorch** implementation of **DGformer**. 



### Requirements
The codebase is implemented in Python 3.8.2. package versions used for development are just below.
```
networkx          2.4
numpy             1.15.4
torch             1.6
```
### Datasets
<p align="justify">
There are two dynamic email networks, called as ENRON and RADOSLAW, in which the nodes and edges represent the users and the e-mails.
FB-FORUM denotes a forum network of the same online community, where edges are the activities participated by two individuals.
    SFHH and INVS are the human contact network, which consists of persons and the real-world contact between two people.<br>
http://networkrepository.com
</p>


Every .edges file has the following structure:

```javascript
from_node to_node weight timestamp
48 		  13 	  1 	 926389620
67        13      1      926418960
67        13      1      926418960
...
```
