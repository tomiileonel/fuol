import os
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

class MongoDBLake:
    """
    Data Lake documental para almacenar perfiles 360 de los partidos.
    Utiliza MongoDB para guardar objetos JSON anidados masivos sin restricciones de esquema rígidas.
    """
    def __init__(self, connection_string=None, database_name="fuol_lake", collection_name="predictions"):
        # Fallback to localhost if no connection string is provided
        self.uri = connection_string or os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
        self.db_name = database_name
        self.collection_name = collection_name
        
        self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        
        # Test connection gracefully
        try:
            self.client.admin.command('ping')
            self.connected = True
            print(f"[MongoDB] Conectado exitosamente a Data Lake ({self.db_name}.{self.collection_name})")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self.connected = False
            print(f"[MongoDB ERROR] No se pudo conectar a MongoDB en {self.uri}. Asegúrate de que el servicio esté corriendo.")

    def save_match_profile(self, master_document: dict) -> bool:
        """
        Guarda (o actualiza) el documento maestro de un partido en el Data Lake.
        Utiliza el 'match_id' para evitar duplicados mediante un upsert.
        """
        if not self.connected:
            print("[MongoDB WARNING] Operación omitida: No hay conexión a base de datos.")
            return False
            
        if "match_id" not in master_document:
            print("[MongoDB ERROR] El documento maestro debe contener un 'match_id'.")
            return False
            
        # Añadir timestamp de inserción si no existe
        if "timestamp" not in master_document:
            master_document["timestamp"] = datetime.now().isoformat()
            
        try:
            # Hacemos un upsert para sobreescribir si ya generamos la misma predicción antes
            result = self.collection.update_one(
                {"match_id": master_document["match_id"]},
                {"$set": master_document},
                upsert=True
            )
            
            if result.upserted_id:
                print(f"[MongoDB] Nuevo perfil creado para {master_document['match_id']} (ID: {result.upserted_id})")
            else:
                print(f"[MongoDB] Perfil actualizado para {master_document['match_id']}")
                
            return True
            
        except Exception as e:
            print(f"[MongoDB ERROR] Fallo al guardar en Data Lake: {e}")
            return False
