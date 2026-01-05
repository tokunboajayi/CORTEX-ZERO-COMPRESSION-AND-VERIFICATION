import re
import json
import zlib
import io

class JSONSplitter:
    """
    Decomposes a stream of similar JSON objects into:
    1. A Schema/Template Stream (The Structure)
    2. Multiple Data Streams (The Variables)
    """
    def __init__(self):
        # We assume a fixed schema for this MVP, or learn simple templates
        self.template_registry = {} # {template_str: template_id}
        self.streams = {
            "structure": [], # [template_id, ...]
            "ints": [],
            "floats": [],
            "strings": [],
            "bools": []
        }
    
    def _tokenize_json(self, obj):
        """
        Recursively traverse JSON, extract values, replace with placeholders.
        Returns: (template_string, list_of_values)
        """
        # Simple flat version for MVP
        # In prod, this would be a proper tree traversal
        
        if isinstance(obj, dict):
            keys = sorted(obj.keys())
            template_parts = []
            values = []
            
            for k in keys:
                v = obj[k]
                sub_template, sub_values = self._tokenize_json(v)
                template_parts.append(f'"{k}":{sub_template}')
                values.extend(sub_values)
            
            return "{" + ",".join(template_parts) + "}", values
            
        elif isinstance(obj, list):
            # For lists, we just placeholder the whole thing for now or iterate
            # MVP: Treat list as a complex value or iterate
            template_parts = []
            values = []
            for item in obj:
                sub_t, sub_v = self._tokenize_json(item)
                template_parts.append(sub_t)
                values.extend(sub_v)
            return "[" + ",".join(template_parts) + "]", values
            
        elif isinstance(obj, bool):
            return "<BOOL>", [(type(obj), obj)]
        elif isinstance(obj, int):
            return "<INT>", [(type(obj), obj)]
        elif isinstance(obj, float):
            return "<FLOAT>", [(type(obj), obj)]
        elif isinstance(obj, str):
            # Heuristic: Short ID-like strings are values, long ones maybe distinct?
            # For now, all strings are values
            return "<STR>", [(type(obj), obj)]
        elif obj is None:
            return "null", []
        
        return "?", []

    def ingest(self, json_line):
        try:
            obj = json.loads(json_line)
            template, values = self._tokenize_json(obj)
            
            # Register Template
            if template not in self.template_registry:
                self.template_registry[template] = len(self.template_registry)
            
            template_id = self.template_registry[template]
            
            # Append to streams
            self.streams["structure"].append(template_id)
            
            for v_type, v_val in values:
                if v_type == int:
                    self.streams["ints"].append(v_val)
                elif v_type == float:
                    self.streams["floats"].append(v_val)
                elif v_type == str:
                    self.streams["strings"].append(v_val)
                elif v_type == bool:
                    self.streams["bools"].append(1 if v_val else 0)
                    
        except json.JSONDecodeError:
            pass # Skip bad lines

    def get_stats(self):
        return {
            "templates": len(self.template_registry),
            "n_ints": len(self.streams["ints"]),
            "n_strs": len(self.streams["strings"])
        }

    def compress_streams(self):
        """
        Compresses the separated streams individually using Zlib (DEFLATE).
        This simulates the 'Entropy Coding' phase.
        """
        compressed_size = 0
        
        # 1. Compress Structure (Int8 IDs)
        # In a real ERT, we'd use a specific entropy coder for these IDs
        struct_bytes = bytes(self.streams["structure"]) # Assuming < 256 templates
        c_struct = zlib.compress(struct_bytes)
        compressed_size += len(c_struct)
        
        # 2. Compress Ints (Delta Encode -> Zlib)
        # Delta encoding is crucial for timestamps/counters
        ints = self.streams["ints"]
        if ints:
            deltas = [ints[0]] + [ints[i] - ints[i-1] for i in range(1, len(ints))]
            # Simple packing: 4 bytes per int for MVP
            delta_bytes = b"".join(d.to_bytes(4, 'big', signed=True) for d in deltas)
            c_ints = zlib.compress(delta_bytes)
            compressed_size += len(c_ints)
        
        # 3. Compress Strings (Concat with separator -> Zlib)
        # In real ERT, we'd use Front Coding or Dict
        if self.streams["strings"]:
            # Use unit separator
            str_data = "\x1F".join(self.streams["strings"]).encode('utf-8')
            c_strs = zlib.compress(str_data)
            compressed_size += len(c_strs)
            
        # 4. Floats/Bools (Raw pack -> Zlib)
        # ... skipped for MVP brevity ...
        
        # 5. Templates Dictionary (Must be stored!)
        dict_str = json.dumps(self.template_registry).encode('utf-8')
        c_dict = zlib.compress(dict_str)
        compressed_size += len(c_dict)
        
        return compressed_size
