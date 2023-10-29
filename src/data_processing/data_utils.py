# Purpose: Utility functions for data processing on FAMuS release format
import json

def famusInstance2ModifiedReportwithTrigger(instance, trigger_tag='event'):
    """
    Given famus release instance,
    output a string of passage sentences with the span highlighted in a trigger_tag
    """
    new_instance = {}

    trigger_span = instance['report_dict']['frame-trigger-span']

    trigger_token_text = trigger_span[0]
    trigger_char_start_idx = trigger_span[1]
    trigger_token_start_idx = trigger_span[3]
    trigger_token_end_idx = trigger_span[4] 
    passage_tokens = instance['report_dict']['doctext-tok']

    # Updated fields
    # we add 1 to the end index because the end index is inclusive
    modified_trigger_tokens = [f"<{trigger_tag}>"] + \
                        passage_tokens[trigger_token_start_idx:trigger_token_end_idx+1] + \
                        [f"</{trigger_tag}>"] 
    modified_tokens = passage_tokens[:trigger_token_start_idx] + \
                        modified_trigger_tokens + \
                        passage_tokens[trigger_token_end_idx+1:]
    
    # the two +1's are for the spaces
    new_trigger_char_length = len(f"<{trigger_tag}>") + \
                                1 + \
                                len(trigger_token_text) + \
                                1 + \
                                len(f"</{trigger_tag}>") 

    # the +2 addition is for the two trigger <event> and </event>  
    # the -1 is because the end index is inclusive      
    modified_trigger_span = [" ".join(modified_trigger_tokens),
                        trigger_char_start_idx,
                        trigger_char_start_idx + new_trigger_char_length - 1,
                        trigger_token_start_idx,
                        trigger_token_end_idx+2,
                        '']
                        

    new_instance = {'doctext': " ".join(modified_tokens),
                    'doctext-tok': modified_tokens,
                    'frame-trigger-span': modified_trigger_span,
                    'colored-doctext': famusInstance2coloredReportText(instance)}
    
    # sanity check: the trigger string should match the span constructed from char indices
    string_from_char_indices = new_instance['doctext'][new_instance['frame-trigger-span'][1]:new_instance['frame-trigger-span'][2]+1]
    assert string_from_char_indices == new_instance['frame-trigger-span'][0]

    return new_instance


def famusInstance2coloredReportText(instance):
    """
    Given famus annotation instance, output
    string of passage sentences with the span highlighted
    in yellow color (Only shows if it is printed)
    """
    from termcolor import colored
    all_tokens = []
    token_intermediate = False
    frame_span = instance['report_dict']['frame-trigger-span']
    token_start_idx = frame_span[3]
    token_end_idx = frame_span[4]
    
    for token_idx, token in enumerate(instance['report_dict']['doctext-tok']):
        if token_idx == token_start_idx:
            all_tokens.append(colored(token, 'yellow'))
            token_length = token_end_idx - token_start_idx + 1
            token_intermediate = True
            countdown_token = token_length
        elif token_intermediate:
            countdown_token -= 1
            if countdown_token:
                all_tokens.append(colored(token, 'yellow'))
            else:
                all_tokens.append(token)
                token_intermediate = False
        else:
            all_tokens.append(token)
            
    return " ".join(all_tokens)


def exportList2Jsonl(list_of_dicts, 
                     output_path):
    with open(output_path, "w") as f:
        for instance in list_of_dicts:
            f.write(json.dumps(instance) + "\n")

def loadJsonl(input_path):
    with open(input_path) as f:
        return [json.loads(line) for line in f]
    