"""
Multi-Hop Retrieval and Graph Reasoning System

This module provides advanced reasoning capabilities through:
- Document relationship graph construction
- Entity linking and recognition
- Multi-hop reasoning paths
- Follow-up question generation
- Graph-based context aggregation
"""

import logging
import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from datetime import datetime
import networkx as nx

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Types of relationships between documents/entities"""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    ENTITY_COREFERENCE = "entity_coreference"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    CAUSAL_RELATIONSHIP = "causal_relationship"
    HIERARCHICAL = "hierarchical"
    REFERENCE = "reference"
    TOPIC_SIMILARITY = "topic_similarity"
    AUTHORSHIP = "authorship"
    CITATION = "citation"


class EntityType(Enum):
    """Types of entities to extract and link"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    MONEY = "money"
    PRODUCT = "product"
    CONCEPT = "concept"
    METRIC = "metric"
    PROCESS = "process"
    TECHNOLOGY = "technology"


@dataclass
class Entity:
    """Represents an extracted entity with metadata"""
    text: str
    entity_type: EntityType
    confidence: float
    canonical_form: str
    aliases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_ids: Set[str] = field(default_factory=set)
    
    def __hash__(self):
        return hash(f"{self.canonical_form}:{self.entity_type.value}")


@dataclass
class DocumentRelationship:
    """Represents a relationship between two documents"""
    source_doc_id: str
    target_doc_id: str
    relationship_type: RelationshipType
    strength: float
    evidence: str
    entities: List[Entity] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningPath:
    """Represents a multi-hop reasoning path through the document graph"""
    start_doc_id: str
    end_doc_id: str
    path_documents: List[str]
    relationships: List[DocumentRelationship]
    total_strength: float
    reasoning_chain: List[str]
    supporting_entities: List[Entity]
    confidence: float


@dataclass
class FollowUpQuestion:
    """Generated follow-up question for deeper exploration"""
    question: str
    reasoning: str
    target_entities: List[Entity]
    expected_answer_type: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class EntityExtractor:
    """Advanced entity extraction with domain awareness"""
    
    def __init__(self):
        # Define entity patterns for different types
        self.entity_patterns = {
            EntityType.PERSON: [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
                r'\b(?:Mr|Mrs|Ms|Dr|CEO|CTO|CFO|President|Director)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            ],
            EntityType.ORGANIZATION: [
                r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\s+(?:Inc|Corp|LLC|Ltd|Company|Corporation|Group|Associates)\b',
                r'\b(?:Department|Division|Team|Unit)\s+of\s+[A-Z][a-z]+\b'
            ],
            EntityType.LOCATION: [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\b',  # City, State
                r'\b(?:in|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
            ],
            EntityType.DATE: [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b',
                r'\b(?:Q[1-4]|[Qq]uarter\s*[1-4])\s*\d{4}\b'
            ],
            EntityType.MONEY: [
                r'\$[\d,]+(?:\.\d{2})?\b',
                r'\b\d+(?:\.\d+)?\s*(?:million|billion|thousand|M|B|K)\s*(?:dollars?|USD)?\b'
            ],
            EntityType.METRIC: [
                r'\b(?:KPI|ROI|EBITDA|P&L|revenue|profit|margin|growth|efficiency|productivity)\b',
                r'\b\d+(?:\.\d+)?%\b',
                r'\b\d+(?:\.\d+)?\s*(?:users?|customers?|employees?|sales?)\b'
            ],
            EntityType.PRODUCT: [
                r'\b[A-Z][a-zA-Z]*\s*(?:v?\d+(?:\.\d+)*|Version\s*\d+)\b',
                r'\b(?:software|platform|system|tool|application|service)\s+[A-Z][a-z]+\b'
            ],
            EntityType.TECHNOLOGY: [
                r'\b(?:API|REST|GraphQL|JSON|XML|HTTP|HTTPS|SQL|NoSQL|AI|ML|blockchain)\b',
                r'\b[A-Z][a-z]+(?:JS|Script|DB|SQL)\b'
            ],
            EntityType.PROCESS: [
                r'\b(?:workflow|process|procedure|method|approach|strategy|framework)\s+[a-z]+\b',
                r'\b[A-Z][a-z]+\s+(?:process|workflow|procedure|method)\b'
            ]
        }
        
        # Entity normalization rules
        self.normalization_rules = {
            EntityType.PERSON: lambda x: self._normalize_person_name(x),
            EntityType.ORGANIZATION: lambda x: self._normalize_organization(x),
            EntityType.DATE: lambda x: self._normalize_date(x),
            EntityType.MONEY: lambda x: self._normalize_money(x),
            EntityType.METRIC: lambda x: x.lower().strip()
        }
        
        # Entity aliases for linking
        self.entity_aliases = {
            'ceo': ['chief executive officer', 'chief executive', 'ceo'],
            'cto': ['chief technology officer', 'chief technical officer', 'cto'],
            'cfo': ['chief financial officer', 'cfo'],
            'roi': ['return on investment', 'roi', 'return'],
            'kpi': ['key performance indicator', 'performance indicator', 'kpi'],
            'api': ['application programming interface', 'programming interface', 'api']
        }
    
    def extract_entities(self, text: str, document_id: str) -> List[Entity]:
        """Extract entities from text with confidence scoring"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entity_text = match.group().strip()
                    
                    # Skip very short or common entities
                    if len(entity_text) < 2 or entity_text.lower() in ['the', 'and', 'or', 'but']:
                        continue
                    
                    # Calculate confidence based on pattern strength and context
                    confidence = self._calculate_entity_confidence(entity_text, entity_type, text, match)
                    
                    if confidence > 0.3:  # Threshold for inclusion
                        # Normalize entity
                        canonical_form = self._normalize_entity(entity_text, entity_type)
                        
                        # Get aliases
                        aliases = self._get_entity_aliases(canonical_form, entity_type)
                        
                        entity = Entity(
                            text=entity_text,
                            entity_type=entity_type,
                            confidence=confidence,
                            canonical_form=canonical_form,
                            aliases=aliases,
                            document_ids={document_id},
                            metadata={
                                'position': match.start(),
                                'context': text[max(0, match.start()-50):match.end()+50]
                            }
                        )
                        
                        entities.append(entity)
        
        # Remove duplicates and merge similar entities
        return self._merge_duplicate_entities(entities)
    
    def _calculate_entity_confidence(self, entity_text: str, entity_type: EntityType, 
                                   full_text: str, match: re.Match) -> float:
        """Calculate confidence score for entity extraction"""
        base_confidence = 0.5
        
        # Boost confidence for entities in specific contexts
        context_window = full_text[max(0, match.start()-100):match.end()+100].lower()
        
        if entity_type == EntityType.PERSON:
            # Boost if preceded by title or role
            if re.search(r'(?:mr|mrs|ms|dr|ceo|director|manager|president)\s*\.?\s*$', 
                        full_text[:match.start()].lower()):
                base_confidence += 0.3
        
        elif entity_type == EntityType.ORGANIZATION:
            # Boost if contains legal entity indicators
            if re.search(r'\b(?:inc|corp|llc|ltd|company|corporation)\b', entity_text.lower()):
                base_confidence += 0.2
        
        elif entity_type == EntityType.METRIC:
            # Boost if in numerical context
            if re.search(r'\d+(?:\.\d+)?', context_window):
                base_confidence += 0.2
        
        # Penalize very short entities
        if len(entity_text) < 4:
            base_confidence -= 0.1
        
        # Boost if entity appears multiple times
        occurrences = full_text.lower().count(entity_text.lower())
        if occurrences > 1:
            base_confidence += min(0.2, occurrences * 0.05)
        
        return min(base_confidence, 1.0)
    
    def _normalize_entity(self, entity_text: str, entity_type: EntityType) -> str:
        """Normalize entity to canonical form"""
        if entity_type in self.normalization_rules:
            return self.normalization_rules[entity_type](entity_text)
        return entity_text.strip()
    
    def _normalize_person_name(self, name: str) -> str:
        """Normalize person name"""
        # Remove titles and normalize capitalization
        name = re.sub(r'^(?:Mr|Mrs|Ms|Dr)\.?\s*', '', name, flags=re.IGNORECASE)
        return ' '.join(word.capitalize() for word in name.split())
    
    def _normalize_organization(self, org: str) -> str:
        """Normalize organization name"""
        # Standardize legal entity suffixes
        org = re.sub(r'\b(Corporation|Corp)\b', 'Corp', org, flags=re.IGNORECASE)
        org = re.sub(r'\b(Incorporated|Inc)\b', 'Inc', org, flags=re.IGNORECASE)
        return org.strip()
    
    def _normalize_date(self, date: str) -> str:
        """Normalize date format"""
        # Try to convert to standard format
        date = date.strip()
        
        # Handle quarter formats
        quarter_match = re.match(r'([Qq]?[1-4])\s*(\d{4})', date)
        if quarter_match:
            return f"Q{quarter_match.group(1)[-1]} {quarter_match.group(2)}"
        
        return date
    
    def _normalize_money(self, money: str) -> str:
        """Normalize monetary amounts"""
        # Extract numeric value and convert to standard format
        money = money.replace(',', '')
        
        # Handle abbreviated forms
        if 'K' in money:
            money = money.replace('K', '000')
        elif 'M' in money:
            money = money.replace('M', '000000')
        elif 'B' in money:
            money = money.replace('B', '000000000')
        
        return money
    
    def _get_entity_aliases(self, canonical_form: str, entity_type: EntityType) -> List[str]:
        """Get aliases for entity linking"""
        canonical_lower = canonical_form.lower()
        aliases = []
        
        # Check predefined aliases
        for alias_key, alias_list in self.entity_aliases.items():
            if canonical_lower in alias_list or any(alias in canonical_lower for alias in alias_list):
                aliases.extend(alias_list)
        
        # Generate automatic aliases for certain types
        if entity_type == EntityType.PERSON:
            # Add initials and short forms
            parts = canonical_form.split()
            if len(parts) >= 2:
                aliases.append(f"{parts[0][0]}. {parts[-1]}")  # First initial + last name
                aliases.append(parts[0])  # First name only
        
        elif entity_type == EntityType.ORGANIZATION:
            # Add acronym if possible
            words = [word for word in canonical_form.split() 
                    if word not in ['Inc', 'Corp', 'LLC', 'Ltd', 'Company', 'Corporation']]
            if len(words) >= 2:
                acronym = ''.join(word[0].upper() for word in words)
                aliases.append(acronym)
        
        return list(set(aliases))
    
    def _merge_duplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge duplicate or very similar entities"""
        merged_entities = {}
        
        for entity in entities:
            # Create a key for grouping similar entities
            key = f"{entity.canonical_form.lower()}:{entity.entity_type.value}"
            
            if key in merged_entities:
                # Merge with existing entity
                existing = merged_entities[key]
                existing.document_ids.update(entity.document_ids)
                existing.confidence = max(existing.confidence, entity.confidence)
                
                # Merge aliases
                existing.aliases = list(set(existing.aliases + entity.aliases))
            else:
                merged_entities[key] = entity
        
        return list(merged_entities.values())


class DocumentGraphBuilder:
    """Builds relationship graph between documents based on entities and content"""
    
    def __init__(self, entity_extractor: EntityExtractor):
        self.entity_extractor = entity_extractor
        self.document_graph = nx.Graph()
        self.entity_document_index = defaultdict(set)  # entity -> set of document_ids
        self.document_entities = defaultdict(list)     # document_id -> list of entities
    
    def add_document(self, document_id: str, content: str, metadata: Dict[str, Any] = None):
        """Add document to the graph with entity extraction"""
        # Extract entities from document
        entities = self.entity_extractor.extract_entities(content, document_id)
        
        # Add document node to graph
        self.document_graph.add_node(
            document_id,
            content_length=len(content),
            entities=len(entities),
            metadata=metadata or {}
        )
        
        # Index entities
        for entity in entities:
            self.entity_document_index[entity.canonical_form].add(document_id)
            self.document_entities[document_id].append(entity)
        
        # Find and add relationships with existing documents
        self._find_document_relationships(document_id, entities, content)
    
    def _find_document_relationships(self, new_doc_id: str, new_entities: List[Entity], 
                                   new_content: str):
        """Find relationships between new document and existing documents"""
        
        # Find documents that share entities
        related_docs = set()
        entity_overlaps = defaultdict(list)
        
        for entity in new_entities:
            related_docs.update(self.entity_document_index[entity.canonical_form])
            
            for doc_id in self.entity_document_index[entity.canonical_form]:
                if doc_id != new_doc_id:
                    entity_overlaps[doc_id].append(entity)
        
        # Create relationships based on entity overlap
        for related_doc_id, shared_entities in entity_overlaps.items():
            if related_doc_id == new_doc_id:
                continue
            
            # Calculate relationship strength
            strength = self._calculate_relationship_strength(
                new_doc_id, related_doc_id, shared_entities
            )
            
            if strength > 0.1:  # Threshold for relationship creation
                # Determine relationship type
                rel_type = self._determine_relationship_type(
                    shared_entities, new_content, 
                    self._get_document_content(related_doc_id)
                )
                
                # Create relationship
                relationship = DocumentRelationship(
                    source_doc_id=new_doc_id,
                    target_doc_id=related_doc_id,
                    relationship_type=rel_type,
                    strength=strength,
                    evidence=f"Shared entities: {[e.canonical_form for e in shared_entities]}",
                    entities=shared_entities,
                    metadata={'shared_entity_count': len(shared_entities)}
                )
                
                # Add edge to graph
                self.document_graph.add_edge(
                    new_doc_id, related_doc_id,
                    relationship=relationship,
                    weight=strength
                )
    
    def _calculate_relationship_strength(self, doc1_id: str, doc2_id: str, 
                                       shared_entities: List[Entity]) -> float:
        """Calculate strength of relationship between two documents"""
        if not shared_entities:
            return 0.0
        
        # Base strength from entity overlap
        strength = min(len(shared_entities) * 0.2, 0.8)
        
        # Boost for high-confidence entities
        confidence_boost = np.mean([e.confidence for e in shared_entities]) * 0.2
        strength += confidence_boost
        
        # Boost for rare entities (entities that appear in fewer documents)
        rarity_boost = 0.0
        for entity in shared_entities:
            entity_doc_count = len(self.entity_document_index[entity.canonical_form])
            if entity_doc_count <= 3:  # Rare entity
                rarity_boost += 0.1
        
        strength += min(rarity_boost, 0.3)
        
        # Penalize if documents are very different in size
        doc1_entities = len(self.document_entities[doc1_id])
        doc2_entities = len(self.document_entities[doc2_id])
        
        if doc1_entities > 0 and doc2_entities > 0:
            size_ratio = min(doc1_entities, doc2_entities) / max(doc1_entities, doc2_entities)
            if size_ratio < 0.3:  # Very different sizes
                strength *= 0.8
        
        return min(strength, 1.0)
    
    def _determine_relationship_type(self, shared_entities: List[Entity], 
                                   content1: str, content2: str) -> RelationshipType:
        """Determine the type of relationship based on shared entities and content"""
        
        # Check for temporal indicators
        if any(e.entity_type == EntityType.DATE for e in shared_entities):
            # Look for temporal sequence indicators
            if any(word in content1.lower() + content2.lower() 
                  for word in ['before', 'after', 'following', 'previous', 'next']):
                return RelationshipType.TEMPORAL_SEQUENCE
        
        # Check for causal indicators
        causal_words = ['because', 'due to', 'resulted in', 'caused', 'led to', 'impact']
        if any(word in content1.lower() + content2.lower() for word in causal_words):
            return RelationshipType.CAUSAL_RELATIONSHIP
        
        # Check for hierarchical indicators
        hierarchy_words = ['part of', 'belongs to', 'under', 'division of', 'department']
        if any(word in content1.lower() + content2.lower() for word in hierarchy_words):
            return RelationshipType.HIERARCHICAL
        
        # Check for reference indicators
        ref_words = ['see', 'refer to', 'mentioned in', 'according to', 'as stated in']
        if any(word in content1.lower() + content2.lower() for word in ref_words):
            return RelationshipType.REFERENCE
        
        # Check entity types for specific relationships
        entity_types = [e.entity_type for e in shared_entities]
        
        if EntityType.PERSON in entity_types and EntityType.ORGANIZATION in entity_types:
            return RelationshipType.ENTITY_COREFERENCE
        
        # Default to semantic similarity
        return RelationshipType.SEMANTIC_SIMILARITY
    
    def _get_document_content(self, doc_id: str) -> str:
        """Get document content (placeholder - would integrate with document store)"""
        # In a real implementation, this would fetch from document store
        return ""
    
    def find_multi_hop_paths(self, start_doc_id: str, target_entities: List[Entity], 
                           max_hops: int = 3) -> List[ReasoningPath]:
        """Find multi-hop reasoning paths from start document to related information"""
        
        if start_doc_id not in self.document_graph:
            return []
        
        reasoning_paths = []
        
        # Use BFS to find paths within max_hops
        visited = set()
        queue = deque([(start_doc_id, [], [], 1.0)])  # (current_doc, path, relationships, strength)
        
        while queue:
            current_doc, path, relationships, current_strength = queue.popleft()
            
            if len(path) > max_hops:
                continue
            
            if current_doc in visited and len(path) > 0:
                continue
            
            visited.add(current_doc)
            
            # Check if current document contains target entities
            current_entities = self.document_entities.get(current_doc, [])
            matching_entities = []
            
            for target_entity in target_entities:
                for doc_entity in current_entities:
                    if (doc_entity.canonical_form.lower() == target_entity.canonical_form.lower() or
                        target_entity.canonical_form.lower() in [alias.lower() for alias in doc_entity.aliases]):
                        matching_entities.append(doc_entity)
            
            if matching_entities and len(path) > 0:  # Found target entities via multi-hop
                reasoning_chain = self._build_reasoning_chain(relationships)
                
                reasoning_path = ReasoningPath(
                    start_doc_id=start_doc_id,
                    end_doc_id=current_doc,
                    path_documents=path + [current_doc],
                    relationships=relationships,
                    total_strength=current_strength,
                    reasoning_chain=reasoning_chain,
                    supporting_entities=matching_entities,
                    confidence=min(current_strength, 1.0)
                )
                
                reasoning_paths.append(reasoning_path)
            
            # Explore neighbors
            if len(path) < max_hops:
                for neighbor in self.document_graph.neighbors(current_doc):
                    if neighbor not in visited:
                        edge_data = self.document_graph.edges[current_doc, neighbor]
                        relationship = edge_data.get('relationship')
                        edge_weight = edge_data.get('weight', 0.5)
                        
                        new_strength = current_strength * edge_weight
                        if new_strength > 0.1:  # Minimum threshold to continue
                            queue.append((
                                neighbor,
                                path + [current_doc],
                                relationships + [relationship],
                                new_strength
                            ))
        
        # Sort by confidence and return top paths
        reasoning_paths.sort(key=lambda p: p.confidence, reverse=True)
        return reasoning_paths[:5]
    
    def _build_reasoning_chain(self, relationships: List[DocumentRelationship]) -> List[str]:
        """Build human-readable reasoning chain from relationships"""
        chain = []
        
        for rel in relationships:
            if rel.relationship_type == RelationshipType.ENTITY_COREFERENCE:
                chain.append(f"Documents are connected through shared entities: {rel.evidence}")
            elif rel.relationship_type == RelationshipType.TEMPORAL_SEQUENCE:
                chain.append(f"Documents are temporally related: {rel.evidence}")
            elif rel.relationship_type == RelationshipType.CAUSAL_RELATIONSHIP:
                chain.append(f"Documents show causal relationship: {rel.evidence}")
            elif rel.relationship_type == RelationshipType.REFERENCE:
                chain.append(f"One document references the other: {rel.evidence}")
            else:
                chain.append(f"Documents are semantically related: {rel.evidence}")
        
        return chain
    
    def get_document_neighborhood(self, doc_id: str, radius: int = 2) -> Dict[str, Any]:
        """Get the local neighborhood of a document in the graph"""
        if doc_id not in self.document_graph:
            return {'nodes': [], 'edges': [], 'entities': []}
        
        # Get subgraph within radius
        subgraph_nodes = set([doc_id])
        current_level = {doc_id}
        
        for _ in range(radius):
            next_level = set()
            for node in current_level:
                neighbors = set(self.document_graph.neighbors(node))
                next_level.update(neighbors)
            
            subgraph_nodes.update(next_level)
            current_level = next_level
        
        subgraph = self.document_graph.subgraph(subgraph_nodes)
        
        # Format response
        nodes = []
        for node in subgraph.nodes():
            nodes.append({
                'id': node,
                'entities': len(self.document_entities.get(node, [])),
                'metadata': subgraph.nodes[node].get('metadata', {})
            })
        
        edges = []
        for edge in subgraph.edges():
            edge_data = subgraph.edges[edge]
            relationship = edge_data.get('relationship')
            if relationship:
                edges.append({
                    'source': edge[0],
                    'target': edge[1],
                    'type': relationship.relationship_type.value,
                    'strength': relationship.strength,
                    'evidence': relationship.evidence
                })
        
        return {
            'center_document': doc_id,
            'nodes': nodes,
            'edges': edges,
            'entities': [
                {
                    'text': e.canonical_form,
                    'type': e.entity_type.value,
                    'confidence': e.confidence
                }
                for e in self.document_entities.get(doc_id, [])
            ]
        }
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the document graph"""
        return {
            'total_documents': self.document_graph.number_of_nodes(),
            'total_relationships': self.document_graph.number_of_edges(),
            'total_entities': len(self.entity_document_index),
            'average_entities_per_doc': np.mean([len(entities) for entities in self.document_entities.values()]) if self.document_entities else 0,
            'most_connected_documents': [
                {'doc_id': node, 'connections': degree}
                for node, degree in sorted(self.document_graph.degree(), key=lambda x: x[1], reverse=True)[:5]
            ],
            'most_common_entities': [
                {'entity': entity, 'document_count': len(doc_set)}
                for entity, doc_set in sorted(self.entity_document_index.items(), 
                                            key=lambda x: len(x[1]), reverse=True)[:10]
            ]
        }


class FollowUpQuestionGenerator:
    """Generates intelligent follow-up questions based on reasoning paths"""
    
    def __init__(self):
        self.question_templates = {
            EntityType.PERSON: [
                "What is {entity}'s role in {context}?",
                "How does {entity} relate to other people mentioned?",
                "What are {entity}'s key contributions or achievements?"
            ],
            EntityType.ORGANIZATION: [
                "What is {entity}'s relationship with other organizations?",
                "How has {entity} performed over time?",
                "What are {entity}'s main products or services?"
            ],
            EntityType.METRIC: [
                "How has {entity} changed over time?",
                "What factors influence {entity}?",
                "How does {entity} compare to industry benchmarks?"
            ],
            EntityType.PROCESS: [
                "What are the steps involved in {entity}?",
                "Who is responsible for {entity}?",
                "What are the outcomes of {entity}?"
            ]
        }
        
        self.relationship_questions = {
            RelationshipType.TEMPORAL_SEQUENCE: [
                "What happened before this event?",
                "What were the consequences of this sequence?",
                "How long did this process take?"
            ],
            RelationshipType.CAUSAL_RELATIONSHIP: [
                "What were the root causes?",
                "What were the downstream effects?",
                "Could this outcome have been prevented?"
            ],
            RelationshipType.HIERARCHICAL: [
                "Who reports to whom in this structure?",
                "What are the responsibilities at each level?",
                "How is information communicated through this hierarchy?"
            ]
        }
    
    def generate_follow_up_questions(self, reasoning_paths: List[ReasoningPath], 
                                   original_query: str, max_questions: int = 5) -> List[FollowUpQuestion]:
        """Generate follow-up questions based on reasoning paths"""
        questions = []
        
        for path in reasoning_paths[:3]:  # Focus on top 3 paths
            # Generate entity-based questions
            for entity in path.supporting_entities[:2]:  # Top 2 entities per path
                if entity.entity_type in self.question_templates:
                    templates = self.question_templates[entity.entity_type]
                    
                    for template in templates[:2]:  # Max 2 questions per entity
                        question_text = template.format(
                            entity=entity.canonical_form,
                            context="the related documents"
                        )
                        
                        question = FollowUpQuestion(
                            question=question_text,
                            reasoning=f"Based on entity '{entity.canonical_form}' found in reasoning path",
                            target_entities=[entity],
                            expected_answer_type="descriptive",
                            confidence=entity.confidence * 0.8,
                            metadata={
                                'source_path': path.path_documents,
                                'entity_type': entity.entity_type.value
                            }
                        )
                        questions.append(question)
            
            # Generate relationship-based questions
            for relationship in path.relationships:
                if relationship.relationship_type in self.relationship_questions:
                    templates = self.relationship_questions[relationship.relationship_type]
                    
                    for template in templates[:1]:  # One question per relationship type
                        question = FollowUpQuestion(
                            question=template,
                            reasoning=f"Based on {relationship.relationship_type.value} relationship between documents",
                            target_entities=relationship.entities,
                            expected_answer_type="analytical",
                            confidence=relationship.strength * 0.7,
                            metadata={
                                'source_relationship': relationship.relationship_type.value,
                                'evidence': relationship.evidence
                            }
                        )
                        questions.append(question)
        
        # Generate comparative questions if multiple paths exist
        if len(reasoning_paths) > 1:
            comparative_questions = [
                "How do these different perspectives compare?",
                "What are the common themes across these documents?",
                "Which source provides the most comprehensive information?"
            ]
            
            for q_text in comparative_questions:
                question = FollowUpQuestion(
                    question=q_text,
                    reasoning="Based on multiple reasoning paths found",
                    target_entities=[],
                    expected_answer_type="comparative",
                    confidence=0.6,
                    metadata={
                        'path_count': len(reasoning_paths),
                        'question_type': 'comparative'
                    }
                )
                questions.append(question)
        
        # Sort by confidence and return top questions
        questions.sort(key=lambda q: q.confidence, reverse=True)
        return questions[:max_questions]


class GraphReasoningManager:
    """Main manager for graph-based reasoning and multi-hop retrieval"""
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.graph_builder = DocumentGraphBuilder(self.entity_extractor)
        self.follow_up_generator = FollowUpQuestionGenerator()
    
    def add_document_to_graph(self, document_id: str, content: str, 
                            metadata: Dict[str, Any] = None):
        """Add a document to the reasoning graph"""
        self.graph_builder.add_document(document_id, content, metadata)
    
    def perform_multi_hop_reasoning(self, query: str, initial_results: List[Dict[str, Any]],
                                  max_hops: int = 3) -> Dict[str, Any]:
        """Perform multi-hop reasoning starting from initial search results"""
        
        if not initial_results:
            return {
                'reasoning_paths': [],
                'follow_up_questions': [],
                'enhanced_context': [],
                'graph_insights': {}
            }
        
        # Extract entities from the query to use as targets
        query_entities = self.entity_extractor.extract_entities(query, "query")
        
        all_reasoning_paths = []
        enhanced_context = []
        
        # Start multi-hop reasoning from each initial result
        for result in initial_results[:3]:  # Limit to top 3 initial results
            doc_id = result.get('metadata', {}).get('document_id') or result.get('id', str(hash(result.get('content', ''))))
            
            # Find reasoning paths from this document
            paths = self.graph_builder.find_multi_hop_paths(doc_id, query_entities, max_hops)
            all_reasoning_paths.extend(paths)
            
            # Add documents from reasoning paths to enhanced context
            for path in paths[:2]:  # Top 2 paths per initial result
                for path_doc_id in path.path_documents:
                    if path_doc_id != doc_id:  # Don't duplicate initial results
                        # In real implementation, would fetch document content
                        enhanced_context.append({
                            'document_id': path_doc_id,
                            'reasoning_chain': path.reasoning_chain,
                            'confidence': path.confidence,
                            'hop_distance': len(path.path_documents) - 1
                        })
        
        # Generate follow-up questions
        follow_up_questions = self.follow_up_generator.generate_follow_up_questions(
            all_reasoning_paths, query
        )
        
        # Get graph insights
        graph_stats = self.graph_builder.get_graph_statistics()
        
        return {
            'reasoning_paths': [
                {
                    'start_document': path.start_doc_id,
                    'end_document': path.end_doc_id,
                    'path_length': len(path.path_documents),
                    'confidence': path.confidence,
                    'reasoning_chain': path.reasoning_chain,
                    'supporting_entities': [
                        {
                            'entity': e.canonical_form,
                            'type': e.entity_type.value,
                            'confidence': e.confidence
                        }
                        for e in path.supporting_entities
                    ]
                }
                for path in all_reasoning_paths[:10]
            ],
            'follow_up_questions': [
                {
                    'question': q.question,
                    'reasoning': q.reasoning,
                    'confidence': q.confidence,
                    'expected_answer_type': q.expected_answer_type
                }
                for q in follow_up_questions
            ],
            'enhanced_context': enhanced_context[:10],
            'graph_insights': {
                'total_reasoning_paths': len(all_reasoning_paths),
                'average_path_confidence': np.mean([p.confidence for p in all_reasoning_paths]) if all_reasoning_paths else 0,
                'entity_coverage': len(query_entities),
                'graph_stats': graph_stats
            }
        }
    
    def get_entity_insights(self, entity_name: str) -> Dict[str, Any]:
        """Get comprehensive insights about a specific entity across documents"""
        # Find all documents containing this entity
        matching_docs = set()
        entity_info = {}
        
        for doc_id, entities in self.graph_builder.document_entities.items():
            for entity in entities:
                if (entity.canonical_form.lower() == entity_name.lower() or
                    entity_name.lower() in [alias.lower() for alias in entity.aliases]):
                    matching_docs.add(doc_id)
                    entity_info = {
                        'canonical_form': entity.canonical_form,
                        'type': entity.entity_type.value,
                        'aliases': entity.aliases
                    }
                    break
        
        if not matching_docs:
            return {'entity_found': False}
        
        # Get document neighborhood for each matching document
        neighborhoods = []
        for doc_id in list(matching_docs)[:5]:  # Limit to 5 documents
            neighborhood = self.graph_builder.get_document_neighborhood(doc_id, radius=2)
            neighborhoods.append(neighborhood)
        
        return {
            'entity_found': True,
            'entity_info': entity_info,
            'document_count': len(matching_docs),
            'document_ids': list(matching_docs),
            'neighborhoods': neighborhoods,
            'related_entities': self._find_related_entities(entity_name, matching_docs)
        }
    
    def _find_related_entities(self, target_entity: str, document_ids: Set[str]) -> List[Dict[str, Any]]:
        """Find entities that frequently co-occur with the target entity"""
        related_entities = defaultdict(int)
        
        for doc_id in document_ids:
            entities = self.graph_builder.document_entities.get(doc_id, [])
            for entity in entities:
                if entity.canonical_form.lower() != target_entity.lower():
                    related_entities[entity.canonical_form] += 1
        
        # Sort by co-occurrence frequency
        sorted_related = sorted(related_entities.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'entity': entity_name, 'co_occurrence_count': count}
            for entity_name, count in sorted_related[:10]
        ]