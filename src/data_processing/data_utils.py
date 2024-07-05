# Purpose: Utility functions for data processing on FAMuS release format
import json
from termcolor import colored
import random

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

from termcolor import colored
import random

def generate_distinct_colors(n):
    colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
    if n <= len(colors):
        return colors[:n]
    else:
        bg_colors = ['on_red', 'on_green', 'on_yellow', 'on_blue', 'on_magenta', 'on_cyan', 'on_white']
        additional_colors = [f"{fg}_{bg}" for fg in colors for bg in bg_colors if fg != bg.split('_')[1]]
        return colors + random.sample(additional_colors, n - len(colors))

def report_or_source_dict_to_role_highlight_text(famus_report_or_source_dict):
    all_tokens = famus_report_or_source_dict['doctext-tok']
    colored_tokens = all_tokens.copy()

    roles = [role for role in famus_report_or_source_dict['role_annotations'].keys() if role != 'role-spans-indices-in-all-spans']
    colors = generate_distinct_colors(len(roles))
    role_colors = dict(zip(roles, colors))

    # Create a list of all markers with their positions
    markers = []
    for role, spans in famus_report_or_source_dict['role_annotations'].items():
        if role != 'role-spans-indices-in-all-spans':
            for span in spans:
                token_start_idx = span[3]
                token_end_idx = span[4]
                markers.append((token_start_idx, f"{role}_start", role))
                markers.append((token_end_idx + 1, f"{role}_end", role))

    # Sort markers by position, with end markers coming before start markers at the same position
    markers.sort(key=lambda x: (x[0], -ord(x[1][-1])))

    # Apply coloring and insert markers
    offset = 0
    active_roles = []
    for i, (pos, marker, role) in enumerate(markers):
        color = role_colors[role]
        if marker.endswith('_start'):
            active_roles.append(role)
        elif marker.endswith('_end'):
            active_roles.pop()

        # Color the marker
        if '_' in color:
            fg, bg = color.split('_')
            colored_marker = colored(f"[{marker}]", fg, bg)
        else:
            colored_marker = colored(f"[{marker}]", color)

        # Insert the marker
        colored_tokens.insert(pos + offset, colored_marker)
        offset += 1

        # Color the text if this is a start marker
        if marker.endswith('_start'):
            current_color = role_colors[active_roles[-1]]  # Use color of the outermost active role
            next_pos = pos + offset
            end_pos = markers[i+1][0] + offset if i+1 < len(markers) else len(colored_tokens)
            
            while next_pos < end_pos:
                if '_' in current_color:
                    fg, bg = current_color.split('_')
                    colored_tokens[next_pos] = colored(colored_tokens[next_pos], fg, bg)
                else:
                    colored_tokens[next_pos] = colored(colored_tokens[next_pos], current_color)
                next_pos += 1

    return " ".join(colored_tokens)


def famus_instance_to_pretty_text_with_roles(instance):
    """
    Given a FAMuS instance, output a string of report and source sentences with the roles highlighted in different colors
    """
    # String denoting frame 
    string = ""
    string += f"Instance_id: {instance['instance_id']}\n"
    string += f"Frame: {instance['frame']}\n"
    string += f"###########################################################\n"
    string += f"#####   Report Text with trigger: ########\n"
    string += f"###########################################################\n"
    string += famusInstance2ModifiedReportwithTrigger(instance)['colored-doctext'] + "\n"
    string += f"###########################################################\n"
    string += f"#####   Report Text with Highlighted Roles: ########\n"
    string += f"###########################################################\n"
    string += report_or_source_dict_to_role_highlight_text(instance['report_dict']) + "\n"
    string += f"###########################################################\n"
    string += f"#####   Source Text with Highlighted Roles: ########\n"
    string += f"###########################################################\n"
    string += report_or_source_dict_to_role_highlight_text(instance['source_dict']) + "\n"

    return string

def exportList2Jsonl(list_of_dicts, 
                     output_path):
    with open(output_path, "w") as f:
        for instance in list_of_dicts:
            f.write(json.dumps(instance) + "\n")

def loadJsonl(input_path):
    with open(input_path) as f:
        return [json.loads(line) for line in f]
    