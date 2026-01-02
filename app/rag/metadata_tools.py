"""
Metadata-based tools for listing files
Use ChromaDB metadata to list all files of a certain type
"""
import logging
from typing import List, Dict, Set
import chromadb
from chromadb.config import Settings
from pathlib import Path

from app.config import rag_config

logger = logging.getLogger(__name__)


class MetadataLister:
    """
    Fast metadata-based listing (no vector search needed!)
    Query ChromaDB metadata to list all files of a type
    """

    def __init__(self):
        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(
            path=rag_config.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(
            name=rag_config.chroma_collection_name
        )

        logger.info(f"✅ MetadataLister initialized ({self.collection.count()} documents)")

    def list_all_controllers(self) -> List[Dict[str, str]]:
        """
        List ALL controllers in the project
        Fast - uses metadata filtering only

        Returns:
            List of {filename, package, class_name, annotations}
        """
        logger.info("📋 Listing all controllers...")

        # Get ALL documents (no limit)
        results = self.collection.get(
            include=["metadatas"]
        )

        controllers = []
        seen_files = set()

        for metadata in results['metadatas']:
            filename = metadata.get('filename', '')
            class_name = metadata.get('class_name', '')
            annotations = metadata.get('annotations', '')

            # Identify controllers by:
            # 1. Filename ends with "Controller.java"
            # 2. Has @RestController or @Controller annotation
            is_controller = (
                    'Controller.java' in filename or
                    'controller' in filename.lower() or
                    '@RestController' in annotations or
                    '@Controller' in annotations
            )

            if is_controller and filename not in seen_files:
                seen_files.add(filename)
                controllers.append({
                    'filename': filename,
                    'class_name': class_name,
                    'package': metadata.get('package', ''),
                    'annotations': annotations,
                    'source': metadata.get('source', '')
                })

        # Sort by filename
        controllers.sort(key=lambda x: x['filename'])

        logger.info(f"✅ Found {len(controllers)} controllers")
        return controllers

    def list_all_services(self) -> List[Dict[str, str]]:
        """List ALL services"""
        logger.info("📋 Listing all services...")

        results = self.collection.get(include=["metadatas"])

        services = []
        seen_files = set()

        for metadata in results['metadatas']:
            filename = metadata.get('filename', '')
            annotations = metadata.get('annotations', '')

            is_service = (
                    'Service.java' in filename or
                    '@Service' in annotations
            )

            if is_service and filename not in seen_files:
                seen_files.add(filename)
                services.append({
                    'filename': filename,
                    'class_name': metadata.get('class_name', ''),
                    'package': metadata.get('package', ''),
                })

        services.sort(key=lambda x: x['filename'])
        logger.info(f"✅ Found {len(services)} services")
        return services

    def list_all_repositories(self) -> List[Dict[str, str]]:
        """List ALL repositories"""
        logger.info("📋 Listing all repositories...")

        results = self.collection.get(include=["metadatas"])

        repositories = []
        seen_files = set()

        for metadata in results['metadatas']:
            filename = metadata.get('filename', '')
            annotations = metadata.get('annotations', '')

            is_repo = (
                    'Repository.java' in filename or
                    'DAO.java' in filename or
                    '@Repository' in annotations
            )

            if is_repo and filename not in seen_files:
                seen_files.add(filename)
                repositories.append({
                    'filename': filename,
                    'class_name': metadata.get('class_name', ''),
                    'package': metadata.get('package', ''),
                })

        repositories.sort(key=lambda x: x['filename'])
        logger.info(f"✅ Found {len(repositories)} repositories")
        return repositories

    def get_project_structure(self) -> Dict[str, List[str]]:
        """
        Get complete project structure

        Returns:
            Dict of {category: [filenames]}
        """
        logger.info("📂 Building project structure...")

        results = self.collection.get(include=["metadatas"])

        structure = {
            'controllers': [],
            'services': [],
            'repositories': [],
            'models': [],
            'helpers': [],
            'filters': [],
            'config': [],
            'other': []
        }

        seen_files = set()

        for metadata in results['metadatas']:
            filename = metadata.get('filename', '')

            if not filename or filename in seen_files:
                continue

            seen_files.add(filename)

            # Categorize
            if 'Controller.java' in filename:
                structure['controllers'].append(filename)
            elif 'Service.java' in filename or '@Service' in metadata.get('annotations', ''):
                structure['services'].append(filename)
            elif 'Repository.java' in filename or 'DAO.java' in filename:
                structure['repositories'].append(filename)
            elif any(x in filename for x in ['Model.java', 'Entity.java', 'DTO.java']):
                structure['models'].append(filename)
            elif 'Filter.java' in filename or 'Interceptor.java' in filename:
                structure['filters'].append(filename)
            elif 'Config.java' in filename or 'Configuration.java' in filename:
                structure['config'].append(filename)
            elif 'Helper.java' in filename or 'Util.java' in filename:
                structure['helpers'].append(filename)
            else:
                structure['other'].append(filename)

        # Sort each category
        for category in structure:
            structure[category].sort()

        # Log summary
        logger.info(f"✅ Project structure:")
        for category, files in structure.items():
            if files:
                logger.info(f"   {category}: {len(files)} files")

        return structure


# Singleton
_lister_instance = None


def get_lister() -> MetadataLister:
    """Get or create MetadataLister singleton"""
    global _lister_instance

    if _lister_instance is None:
        _lister_instance = MetadataLister()

    return _lister_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    lister = MetadataLister()

    # Test
    print("\n" + "=" * 80)
    print("ALL CONTROLLERS:")
    print("=" * 80)

    controllers = lister.list_all_controllers()
    for i, ctrl in enumerate(controllers, 1):
        print(f"{i}. {ctrl['filename']}")
        if ctrl['class_name']:
            print(f"   Class: {ctrl['class_name']}")
        if ctrl['package']:
            print(f"   Package: {ctrl['package']}")

    print("\n" + "=" * 80)
    print("PROJECT STRUCTURE:")
    print("=" * 80)

    structure = lister.get_project_structure()
    for category, files in structure.items():
        if files:
            print(f"\n{category.upper()} ({len(files)}):")
            for file in files[:10]:  # Show first 10
                print(f"  - {file}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")