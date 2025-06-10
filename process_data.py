import json

def convert_relation_data(input_data, input_index, relation_labels, model_name, fire):
    """
    Convert relation extraction data from original format to target format.
    
    Args:
        input_data: Dictionary with original format
        input_index: Index for instances that decode together
        model_name: Model name to assign
    
    Returns:
        List of two dictionaries in target format
    """
    

    
    # for fire dataset
    if fire:
        subject = ' '.join(input_data['token'][input_data['e1_start']:input_data['e1_end']])
        object_entity = ' '.join(input_data['token'][input_data['e2_start']:input_data['e2_end']])
    else:
        # Extract subject and object spans
        subject = ' '.join(input_data['token'][input_data['subj_start']:input_data['subj_end']+1])
        object_entity = ' '.join(input_data['token'][input_data['obj_start']:input_data['obj_end']+1])
    relation = input_data['relation']
    
    # Create the full context string (tokens joined with spaces)
    tokens_text = ' '.join(input_data['token'])
    
    # Create the question
    question = f"What is the relationship between the '{subject}' and the '{object_entity}'?\n"
    question += f"Choose only ONE from the following options. Options: {relation_labels}\n Answer:"
    
    # First instance: with context
    context_with_question = f"{tokens_text}\n{question}"
    
    
    # Create the two output instances
    instance_with_context = {
        "input_index": input_index,
        "assigned_model": model_name,
        "assigned_process": 0,
        "context_string": context_with_question,
        "assigned_weight": 2,
        "filter_p": 1.0,
        "original_answer": relation,
        "gold_answers": relation
    }
    
    instance_without_context = {
        "input_index": input_index,
        "assigned_model": model_name,
        "assigned_process": 1,
        "context_string": question,
        "assigned_weight": -1,
        "original_answer": relation,
        "gold_answers": relation
    }
    
    return [instance_with_context, instance_without_context]

def process_file(input_file_path, output_file_path, model_name="meta-llama/Meta-Llama-3-8B"):
    """
    Process an entire file of relation extraction data.
    
    Args:
        input_file_path: Path to input JSON file
        output_file_path: Path to output JSONL file
        model_name: Model name to assign
    """
    is_fire_dataset=False
    if 'fire' in input_file_path:
        is_fire_dataset = True
    
    with open(input_file_path, 'r') as f:
        # Read all lines if it's JSONL, or single JSON object
        content = f.read().strip()
        if content.startswith('['):
            # JSON array
            data_list = json.loads(content)
        else:
            # JSONL format
            data_list = [json.loads(line) for line in content.split('\n') if line.strip()]

    relation_labels = list(set([instance['relation'] for instance in data_list]))
    
    with open(output_file_path, 'w') as f:
        input_index = 0
        for data in data_list:
            converted_instances = convert_relation_data(data, input_index, 
                                                        relation_labels=relation_labels, 
                                                        model_name=model_name, 
                                                        fire=is_fire_dataset)
            for instance in converted_instances:
                f.write(json.dumps(instance) + '\n')
            input_index += 1

if __name__ == "__main__":
    # meta-llama/Llama-3.1-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.3, tiiuae/Falcon3-10B-Instruct
    model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
    process_file(input_file_path='data/fire/test.json', 
                 output_file_path='data/fire/nq_test.jsonl', 
                 model_name=model_name)
    # process_file(input_file_path='data/fire/dev.json', 
    #              output_file_path='data/fire/nq_dev.jsonl', 
    #              model_name=model_name)
    
    # process_file(input_file_path='data/biored/test.json', 
    #              output_file_path='data/biored/nq_test.jsonl', 
    #              model_name=model_name)

    process_file(input_file_path='data/refind/test.json', 
                 output_file_path='data/refind/nq_test.jsonl', 
                 model_name=model_name)


    # process_file(input_file_path='data/refind/test.json', 
    #              output_file_path='data/refind/nq_test.jsonl', 
    #              model_name=model_name)




    
#     process_file(input_file_path='data/tacred/test_entred.json', 
#                  output_file_path='data/tacred/nq_test_entred.jsonl', 
#                  model_name="mistralai/Mistral-7B-Instruct-v0.3")

    
#     process_file(input_file_path='data/refind/test_entred.json', 
#                  output_file_path='data/refind/nq_test_entred.jsonl', 
#                  model_name="mistralai/Mistral-7B-Instruct-v0.3")
