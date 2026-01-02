"""
Neo4j graph construction from Java codebase
Maps relationships: Controller->Service->Repository->Entity
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
import javalang
from neo4j import GraphDatabase

from app.config import settings, get_neo4j_auth, get_data_path

logger = logging.getLogger(__name__)


class CodeGraphBuilder:
    """
    Build code relationship graph in Neo4j
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=get_neo4j_auth()
        )

    def close(self):
        self.driver.close()

    def build_graph(self):
        """
        Build complete code graph from Java files
        """
        logger.info("Building code relationship graph...")

        # Clear existing graph
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

        # Find all Java files
        code_dir = get_data_path("codebase")
        java_files = list(code_dir.rglob("*.java"))

        logger.info(f"Found {len(java_files)} Java files")

        for java_file in java_files:
            try:
                self._process_file(java_file)
            except Exception as e:
                logger.error(f"Error processing {java_file}: {e}")

        logger.info("Graph construction complete")

    def _process_file(self, file_path: Path):
        """Process single Java file and extract relationships"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        try:
            tree = javalang.parse.parse(content)
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return

        package = tree.package.name if tree.package else "default"

        # Process each class
        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            class_name = node.name
            annotations = [f"@{ann.name}" for ann in (node.annotations or [])]

            # Determine node type based on annotations
            node_type = self._determine_node_type(annotations)

            # Create node
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (c:Class {name: $name, package: $package})
                    SET c.type = $type,
                        c.annotations = $annotations,
                        c.file = $file
                    """,
                    name=class_name,
                    package=package,
                    type=node_type,
                    annotations=annotations,
                    file=str(file_path)
                )

            # Extract relationships
            self._extract_relationships(class_name, package, node, content)

    def _determine_node_type(self, annotations: List[str]) -> str:
        """Determine class type from annotations"""
        if any(a in annotations for a in ['@RestController', '@Controller']):
            return 'Controller'
        elif '@Service' in annotations:
            return 'Service'
        elif '@Repository' in annotations:
            return 'Repository'
        elif '@Entity' in annotations:
            return 'Entity'
        else:
            return 'Class'

    def _extract_relationships(
            self, class_name: str, package: str, node, content: str
    ):
        """Extract method calls and field dependencies"""
        with self.driver.session() as session:
            # Extract field dependencies (autowired fields)
            for field in node.fields:
                if field.declarators:
                    field_type = field.type.name if hasattr(field.type, 'name') else str(field.type)

                    # Check if field is autowired
                    is_autowired = any(
                        ann.name == 'Autowired'
                        for ann in (field.annotations or [])
                    )

                    if is_autowired:
                        # Create USES relationship
                        session.run(
                            """
                            MATCH (c1:Class {name: $from_class, package: $package})
                            MERGE (c2:Class {name: $to_class})
                            MERGE (c1)-[:USES]->(c2)
                            """,
                            from_class=class_name,
                            package=package,
                            to_class=field_type
                        )

            # Extract method calls (simplified heuristic)
            for method in node.methods:
                # Parse method body for calls (simple regex approach)
                method_start = method.position.line if method.position else 0
                # This is simplified - in production, use AST traversal

    def query_relationships(self, entity_name: str, depth: int = 2) -> List[Dict[str, Any]]:
        """
        Query relationships for given entity

        Args:
            entity_name: Class name to start from
            depth: Maximum path depth

        Returns:
            List of relationship paths
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (start:Class {name: $name})-[*1..$depth]-(related:Class)
                RETURN path,
                       start.name as start_name,
                       related.name as end_name,
                       [r in relationships(path) | type(r)] as rel_types
                LIMIT 20
                """,
                name=entity_name,
                depth=depth
            )

            paths = []
            for record in result:
                paths.append({
                    'start': record['start_name'],
                    'end': record['end_name'],
                    'relationships': record['rel_types']
                })

            return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    builder = CodeGraphBuilder()
    builder.build_graph()
    builder.close()