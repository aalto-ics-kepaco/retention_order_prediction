####
#
# The MIT License (MIT)
#
# Copyright 2017, 2018 Eric Bach <eric.bach@aalto.fi>,
#                      Sandor Szedmak <sandor.szedmak@aalto.fi>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
####

## ##################################################
import csv

import numpy as np
import networkx as nx # should be version >= 2.0

import itertools as it

from collections import OrderedDict
from scipy.stats import rankdata
## ##################################################
## ##################################################
class retention_cls:
  """
  Task: load and process the retention time data
  """

  ## -----------------------------------------
  def __init__(self):
    """
    Task: Initialize the general parameters of this class.
    """
    self.reset()

    return

  def reset(self):
    """
    Task: Reset the class parameters to their initial values,
          i.e. empty the lists and dictionaries, and delete
          the graphs.
    """
    self.lrows=[]    ##data list of list:
                     ## [['node_out','node_in',edgetype,'collection'],...]
    self.dmolecules={}   ## dictionary of the nodes: {node string: node index}
    self.dmolecules_inv={}   ## invert dnode: { node index : node string}
    self.dcollections={} ## dictionary of the collections:
                         ## {collection string: collection index}
    self.dcollections_inv={}   ## invert dcollection
                              ##  { collection index : collection string }
    self.dmolecule2collection={}  ## dictionary of sets of collection
                              ## belonging to molecule:
                              ## { molecule index : {collection index, ....}}
  
    self.dG=None     ## the directed graph of the data by networkx object 
    self.G=None      ## undirected dG

    self.lmaximal_nodes=[]  ## list of nodes with no incoming edge    

    return

  ## ---------------------------------------------
  def load_data_from_file(self, sfile, nheader=1, linclude_collection='__ALL__', linclude_node='__ALL__'):
    """
    Task: to load the data from csv file into self.lrows

    Input:  sfile string, path to the file containing the order information

            nheader integer, number of header lines in the file

            linclude_collection
              list of collection-ids to include in the graph: [collection-id, ...]
              if '__ALL__' ==> all collections are included

            linclude_node
              list of node-ids to include in the graph: [node-id, ...]
              if '__ALL__' ==> all nodes are included
    """
    self.reset()

    if isinstance(linclude_collection,str) and linclude_collection=='__ALL__':
      include_all_collections=True
    else:
      include_all_collections=False

    if isinstance(linclude_node,str) and linclude_node=='__ALL__':
      include_all_nodes=True
    else:
      include_all_nodes=False

    irow=0
    with open(sfile) as csvfile:
      csvreader=csv.reader(csvfile, delimiter=',', quotechar='"')
      for row in csvreader:
        if (irow>=nheader):
          if (include_all_collections or row[3] in linclude_collection) and \
                  (include_all_nodes or row[0] in linclude_node and row[1] in linclude_node):
            self.lrows.append(row)

        irow+=1

    return

  ## ---------------------------------------------
  def load_data_from_target(self, d_target, linclude_collection='__ALL__', linclude_node='__ALL__'):
    """
    Task: To fill the self.lrows list with the order information.
          Thereby the order information are extracted on the fly
          from the given target values.

    Input:  d_target dictionary,
                     2nd level: keys are (node,collection)-tuples and the values
                                are the associated target values.

                     Example:
                      {("mol1","s1"): t_11, ("mol2","s1"): t_21, ...,
                       ("mol1","s2"): t_12, ("mol3","s2"): t_32, ...}

            linclude_collection
              list of collection-ids to include in the graph: [collection-id, ...]
              if '__ALL__' ==> all collections are included

            linclude_node
              list of node-ids to include in the graph: [node-id, ...]
              if '__ALL__' ==> all nodes are included
    """
    self.reset()

    if isinstance(linclude_collection,str) and linclude_collection=='__ALL__':
      include_all_collections=True
    else:
      include_all_collections=False

    if isinstance(linclude_node,str) and linclude_node=='__ALL__':
      include_all_nodes=True
    else:
      include_all_nodes=False


    collections=np.unique([key[1] for key in d_target.keys()])

    for collection in collections:
      # If the current collections should be excluded --> continue
      if not include_all_collections and not collection in linclude_collection:
        continue

      # Get dictionary of ranks and their corresponding nodes
      # 1) Determine the rank if each node based on their target values
      # 2) Build a dictionary (rank) --> [node_1,node_2,...], that
      #    collects the ranks and all nodes corresponding to each rank.
      #    NOTE: Nodes can share a rank.
      # 3) Determine the maximum rank
      d_target_coll={key:value for key,value in d_target.items()
                     if (include_all_nodes or key[0] in linclude_node) and key[1] == collection}
      if len(d_target_coll)==0:
        return

      nodes=np.array(list(zip(*d_target_coll.keys()))[0])
      targets=np.array(list(d_target_coll.values()))
      ranks=rankdata(targets,method="dense") # ave-perf O(n log(n))

      # Sort the ranks and nodes
      sorter=np.argsort(ranks) # ave-perf O(n log(n))
      nodes=nodes[sorter]
      ranks=ranks[sorter]

      d_nodes_coll=OrderedDict()
      max_rank=0 # smallest rank by 'rankdata' is 1
      for node,rank in zip(nodes,ranks): # O(n)
        if rank in d_nodes_coll:          # O(1), worst O(n)
          d_nodes_coll[rank].append(node) # O(1), worst O(n)
        else:
          d_nodes_coll[rank]=[node]     # O(1), worst O(n)

        # Maintain information about the largest rank
        if rank>max_rank:
          max_rank=rank

      for rank_out in range(1,max_rank+1): # {1, 2, ..., max_rank}, O(max_rank)
        node_out=d_nodes_coll[rank_out]

        for rank_in in range(rank_out,max_rank+1): # {current_rank, ..., max_rank}, O(max_rank - current_rank), best O(1)
          # Transitive order information are not considered
          if rank_in>rank_out+1:
            break

          node_in=d_nodes_coll[rank_in]

          if rank_out==rank_in:
            assert(node_out==node_in)

            if len(node_out)<2:
              continue

            for comb in it.combinations(node_out,2):
              self.lrows.append([comb[0],comb[1],0,collection])
              self.lrows.append([comb[1],comb[0],0,collection])
          else:
            assert(not any([x in node_in for x in node_out]))

            for comb in it.product(node_out,node_in):
              self.lrows.append([comb[0],comb[1],1,collection])

    return

  ## ---------------------------------------------
  def make_digraph (self, ireverse=1, iostrict=1):
    """
    Task: to make a directed graph object of networkx from the order data

          The nodes of the graph are tuples: (molecule index, collection index)
          There is an edge between two nodes
              (molecule1,collection1) and (molecule2,collection2)
              if
              1.  molecule1 != molecule2
                  AND molecule1 and molecule2 appears in the same row 
                  AND collection1=collection2

                  edge direction follow molecule order in the rows

              2.  molecule1=molecule2
                  AND collection1!=collection2

                  edge is bidirectional:
                  edge=((molecule1,collection1),(molecule1,collection2))
                      when collection1 index > collection2 index

    Input:   ireverse    =0 only first type of edges are loaded
                         =1 both type 1 and 2 edges are loaded
             iostrict: Should molecules with the same retention time
                       within one system be connected? (include only strict orders)
                         =0 edge-type 0, e.g. same retention time, is included
                         =1 edge-type 0 is ignored
    """

    self.dG=nx.DiGraph()

    imolecule=0
    icollection=0
    ## collect molecule strings and assigne index to them
    ## collect collection strings and assign index to them
    irow=0
    for row in self.lrows:
      if iostrict and int (row[2]) == 0:
        continue

      if row[0] not in self.dmolecules:
        self.dmolecules[row[0]]=imolecule
        imolecule+=1
      if row[1] not in self.dmolecules:
        self.dmolecules[row[1]]=imolecule
        imolecule+=1
      molecule_h=self.dmolecules[row[0]]  ## head node
      molecule_t=self.dmolecules[row[1]]  ## tail node
      if molecule_h==molecule_t:
        print(row)
        
      if row[3] not in self.dcollections:
        self.dcollections[row[3]]=icollection
        icollection+=1
      collection=self.dcollections[row[3]]

      ## add edge
      self.dG.add_edge((molecule_h,collection),(molecule_t,collection), weight = 1)
      ## if molecule_h==2 and collection==1:
      ##   print(irow,row)
      
      ## gather collections to each molecule
      if molecule_h not in self.dmolecule2collection:
        self.dmolecule2collection[molecule_h]=set([])
      self.dmolecule2collection[molecule_h].add(collection)
      if molecule_t not in self.dmolecule2collection:
        self.dmolecule2collection[molecule_t]=set([])
      self.dmolecule2collection[molecule_t].add(collection)
      irow+=1

    if ireverse==1:
      ## add the edges to repeated molecules in collections
      for molecule,scollection in self.dmolecule2collection.items():
        lcollection=[collection for collection in scollection]
        lcollection.sort(reverse=True)
        ncollection=len(lcollection)
        for i in range(ncollection-1):
          for j in range(i+1,ncollection):
            ## add edge
            self.dG.add_edge((molecule,lcollection[i]), (molecule,lcollection[j]), weight = 0)
            self.dG.add_edge((molecule,lcollection[j]), (molecule,lcollection[i]), weight = 0)
            ## print((molecule,lcollection[i]),(molecule,lcollection[j]))

    return

  ## -------------------------------------
  def upper_lower_set_node(self,G,path_algo="dijkstra"):
    """
    Task: to find to upper and lower set (U,L) in directed graph G to all nodes.
      upper set U is the set of nodes from which there is a path to a node N 
      lower set L is the set of nodes to which there is a path from a node N
    Input: G          networkx directed graph

           path_algo  string, which algorithm is used for the pairwise distance
                      calculation:
                      "dijkstra": 'all_pairs_dijkstra_path_length'
                      "shortest"  'all_pairs_shortest_path_length'

    Output:  dnodecut  dictionary
             { node : (upper set of nodes, lower set of nodes)}
             upper(lower) set is given by a dictionary
                         { node : length from shorthes path }
    """

    dnodecut={}
    ## collect pairs of nodes such that one can reach the other
    if path_algo == "dijkstra":
      length=nx.all_pairs_dijkstra_path_length(G, weight = "weight")
    elif path_algo == "shortest":
      length=nx.all_pairs_shortest_path_length(G)
    else:
      raise ValueError ("Invalid path algorithm %s." % path_algo)

    icount=0
    for node1,dchilds in length:
      if node1 not in dnodecut:
        dnodecut[node1]=({},{})
      for node2,dist in dchilds.items():
        if node2 not in dnodecut:
          dnodecut[node2]=({},{})
        if node1==node2:
          continue       ## node is not part of the upper nad lower sets
        ## since there is path from node1 to node2 therefore:
        ## node2 in lower set of node1, 
        dnodecut[node1][1][node2]=dist
        ## node1 in upper set of node2 
        dnodecut[node2][0][node1]=dist
      icount+=1
    
    return(dnodecut)

  ## -------------------------------
  ## General utility methods
  ## -------------------------------
  def invert_dictionary(self,dictionary):
    """
    Task: to invert dictionary {key :value}
          assumptions:
               value is immutable
               the dictionary is bijective between key and value
    Input: source dictionary { key : value}
    Output target dictionary {value : key }
    """

    inv_dictionary={}
    for key,value in dictionary.items():
      inv_dictionary[value]=key

    return(inv_dictionary)