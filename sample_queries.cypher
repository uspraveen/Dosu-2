// Find Complex Functions
// Find the most complex functions
            MATCH (f:Function)
            WHERE f.complexity IS NOT NULL
            RETURN f.name, f.complexity, f.file_path, f.github_url
            ORDER BY f.complexity DESC
            LIMIT 10

// Find Function Calls
// Find functions that call a specific function
            MATCH (caller:Function)-[:CALLS_LOCAL|CALLS_IMPORTED|CALLS_METHOD|CALLS_GLOBAL]->(callee:Function {name: $function_name})
            RETURN caller.name, caller.file_path, caller.github_url

// Go To Function Definition
// Go to definition: Find where a called function is defined
            MATCH (caller_file:File {path: $caller_file_path})
            
            // Priority 1: Local function in same file
            OPTIONAL MATCH (caller_file)<-[:DEFINED_IN]-(local_func:Function {name: $function_name})
            
            // Priority 2: Imported function
            OPTIONAL MATCH (caller_file)-[:IMPORTS]->(imp:Import)-[:RESOLVES_TO]->(imported_func:Function)
            WHERE imported_func.name = $function_name
            
            // Priority 3: Method in class context
            OPTIONAL MATCH (caller_file)<-[:DEFINED_IN]-(caller:Function {name: $caller_function_name})
                          -[:BELONGS_TO]->(class:Class)
                          <-[:BELONGS_TO]-(method:Function {name: $function_name})
            
            // Priority 4: Global search
            OPTIONAL MATCH (global_func:Function {name: $function_name})
            
            WITH 
                local_func, imported_func, method, global_func,
                CASE 
                    WHEN local_func IS NOT NULL THEN 1
                    WHEN imported_func IS NOT NULL THEN 2  
                    WHEN method IS NOT NULL THEN 3
                    ELSE 4
                END as priority
            
            WITH 
                CASE 
                    WHEN local_func IS NOT NULL THEN local_func
                    WHEN imported_func IS NOT NULL THEN imported_func
                    WHEN method IS NOT NULL THEN method 
                    ELSE global_func
                END as definition, priority
            
            WHERE definition IS NOT NULL
            
            RETURN 
                definition.name as function_name,
                definition.file_path as file_path,
                definition.line_start as line_number,
                definition.github_url as github_link,
                definition.docstring as documentation,
                CASE priority
                    WHEN 1 THEN 'local'
                    WHEN 2 THEN 'imported'  
                    WHEN 3 THEN 'method'
                    ELSE 'global'
                END as resolution_type
            ORDER BY priority LIMIT 1

// Find All Usages
// Find all usages: Where is this function/class used?
            MATCH (definition)
            WHERE definition.name = $definition_name
              AND definition.node_type IN ['function', 'class']
            
            // Find enhanced function calls
            OPTIONAL MATCH (caller:Function)-[:CALLS_LOCAL|CALLS_IMPORTED|CALLS_METHOD|CALLS_GLOBAL]->(definition)
            WHERE definition.node_type = 'function'
            
            // Find class inheritance
            OPTIONAL MATCH (child:Class)-[:INHERITS]->(definition)  
            WHERE definition.node_type = 'class'
            
            // Find imports
            OPTIONAL MATCH (imp:Import)-[:RESOLVES_TO]->(definition)
            
            RETURN DISTINCT
                CASE 
                    WHEN caller IS NOT NULL THEN caller.file_path
                    WHEN child IS NOT NULL THEN child.file_path
                    WHEN imp IS NOT NULL THEN imp.file_path
                END as usage_file,
                CASE 
                    WHEN caller IS NOT NULL THEN caller.line_start
                    WHEN child IS NOT NULL THEN child.line_start  
                    WHEN imp IS NOT NULL THEN imp.line_start
                END as usage_line,
                CASE 
                    WHEN caller IS NOT NULL THEN 'function_call'
                    WHEN child IS NOT NULL THEN 'inheritance'
                    WHEN imp IS NOT NULL THEN 'import'
                END as usage_type,
                CASE 
                    WHEN caller IS NOT NULL THEN caller.github_url
                    WHEN child IS NOT NULL THEN child.github_url
                    WHEN imp IS NOT NULL THEN imp.github_url  
                END as github_link

// Find Class Hierarchy
// Find class inheritance hierarchy
            MATCH path = (child:Class)-[:INHERITS*]->(parent:Class)
            WHERE parent.name = $class_name
            RETURN path

// Find Similar Functions
// Find functions similar to a given function (requires embeddings)
            MATCH (target:Function {name: $function_name})
            WHERE target.embedding IS NOT NULL
            
            CALL db.index.vector.queryNodes('code_embeddings', 5, target.embedding)
            YIELD node, score
            WHERE node <> target
            RETURN node.name, node.file_path, node.github_url, score

// Code Metrics
// Get repository code metrics
            MATCH (f:Function)
            WITH count(f) as total_functions,
                 avg(f.complexity) as avg_complexity,
                 max(f.complexity) as max_complexity
            
            MATCH (c:Class)
            WITH total_functions, avg_complexity, max_complexity, count(c) as total_classes
            
            MATCH (file:File)
            RETURN {
                total_functions: total_functions,
                total_classes: total_classes,
                total_files: count(file),
                avg_complexity: round(avg_complexity, 2),
                max_complexity: max_complexity
            } as metrics

