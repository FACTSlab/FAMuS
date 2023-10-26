from typing import Tuple, List
from nltk.corpus import framenet as fn
from bs4 import BeautifulSoup
from bs4.element import Tag
from collections import defaultdict
from stanza.pipeline.core import Pipeline as StanzaPipeline
    
ALL_FRAMES = set([frame.name for frame in fn.frames()])

def frame_to_def(frame: str,
                stanza_pipeline: StanzaPipeline):
    """
    definition of a frame (take only the first sentence)
    """
    doc = stanza_pipeline(fn.frame(frame)['definition'])
    sents = [sentence.text for sentence in doc.sentences]
    # For small errors where sentence boundary splits the first sentence
    # Example: Frame: 'Arriving'
    if len(sents[0].split()) < 5:
        return sents[0] + " " + sents[1]
    return sents[0]


def frame_to_core_roles(frame: str):
    """
    extract a list of all core roles
    """
    ## Added extra roles that are generally frequent
    extra_roles = ['Time', 'Place']

    extra_valid_roles = [role for role in extra_roles if role in fn.frame(frame).FE]

    core_roles = [ role for role, dcts in fn.frame(frame).FE.items() 
                                    if dcts['coreType']=="Core" or dcts['coreType']=="Core-Unexpressed"]

    extra_valid_roles = [role for role in extra_valid_roles if role not in core_roles]

    return core_roles + extra_valid_roles


def simplify_fex_tag(string):
    """
    simplify the html tag to have no value:
    eg: <fex name="Victim">the guardsmen</fex> -> <fex>the guardsmen</fex>
    """
    import re
    string = re.sub(r'<fex name="[^"]+">', r'<fex>', string)
    return string


def make_double_quotes_single(string):
    return string.replace(r'"', "'")


def frame_to_role_def_example(frame: str, role: str):
    """
    Given a frame and its role, get the role definition and role example
    """
    import re
    soup = BeautifulSoup(fn.frame(frame).FE[role]['definitionMarkup'],features="lxml")

    # a role can used inside <fex> tag with either its name or its abbrev
    role_names = [role, fn.frame(frame).FE[role]['abbrev']]
    
    # get definition of role:
    string_def = " "
    for child in list(soup.find("def-root").children):
        if "<ex>" not in str(child):
            string_def += str(child)
        else:
            break
    string_def= string_def.strip()

    # replace "FE" by role in the definition of a role
    string_def = re.sub(r' FE ', r' role ', string_def)
    
    # Get the first example of the role
    examples = soup.findAll("ex")
    if examples:
        example = examples[0]
        # remove the <fex> tag for other roles
        example_str = ""
        for child in example.children:
            if isinstance(child, Tag):
                if child.name == "fex" and child.get("name") not in role_names:
                    # only get contents of roles that are not this role
                    string_contents = [str(c) for c in child.contents]
                    example_str+= " ".join(string_contents)
                elif child.name != "fex":
                    # only get contents of other tags (such as <t> tag)
                    string_contents = [str(c) for c in child.contents]
                    example_str+= " ".join(string_contents)
                else:
                    example_str += str(child)
            else:
                example_str += str(child)    
    else:
        example_str = ""

    return (string_def, simplify_fex_tag(example_str))


def frame_to_frame_example(frame: str):
    """
    Given a frame, get the frame example
    """
    soup = BeautifulSoup(fn.frame(frame)['definitionMarkup'],features="lxml")
    examples = soup.findAll("ex")
    if examples:
        examples_with_contents = [example for example in examples if example.contents]
        if not examples_with_contents:
            return ""
        # get the first example where some contents are present
        example = examples_with_contents[0]
        # remove the <fex> tag for other roles
        example_str = ""
        for child in example.children:
            if isinstance(child, Tag):
                if child.name != "t":
                    # only get contents (and exclude tags are not equal to <t>)
                    string_contents = [str(c) for c in child.contents]
                    example_str+= " ".join(string_contents)
                else:
                    example_str += str(child)
            else:
                example_str += str(child)    
    else:
        example_str = ""
    return example_str


def frame_to_info_dct(frame: str,
                      stanza_pipeline: StanzaPipeline):
    """
    Given an input frame, extract the required info (for rams2 annotation) as JSON dicts
    """
    core_roles = frame_to_core_roles(frame)

    dct = {"frameDefinitions": [make_double_quotes_single(frame_to_def(frame, stanza_pipeline))],

    "frameExamples": [make_double_quotes_single(frame_to_frame_example(frame))],

    'roles':  core_roles,

    'roleDefinitions': [make_double_quotes_single(frame_to_role_def_example(frame, role)[0]) for role in core_roles],

    'roleExamples': [make_double_quotes_single(str(frame_to_role_def_example(frame, role)[1])) for role in core_roles],    
    }

    return dct


def frame_to_llm_prompt_info_dct(frame_name, stanza_nlp):
    """
    Given a frame name, return a dictionary with the following keys:
    - event_type: frame_name
    - event_definition: frame definition with example
    - event_roles: frame roles with definitions and examples
    """
    info_dct = frame_to_info_dct(frame_name, stanza_nlp)
    # Event Roles
    event_def_roles = ""
    for idx, (role, role_def, role_example) in enumerate(zip(info_dct['roles'], 
                                        info_dct['roleDefinitions'], 
                                        info_dct['roleExamples'])):
        event_def_roles += f"{idx+1}. {role}: {role_def}"
        event_def_roles += f"\n{role_example}\n"

    return {'event_type': frame_name,
            'event_definition': info_dct['frameDefinitions'][0],
            'event_roles': event_def_roles}
    

def frame_list_to_info_dct(frame_lst: List[str],
                            stanza_pipeline: StanzaPipeline):
    """
     Given an input list of frames, extract the required info (for rams2 annotation) as JSON dicts
    """
    list_info_dct = defaultdict(list)

    for frame in frame_lst:
        if frame not in ALL_FRAMES:
            continue
        curr_dct = frame_to_info_dct(frame, stanza_pipeline)
        list_info_dct['frameDefinitions'].append(curr_dct['frameDefinitions'][0])
        list_info_dct['frameExamples'].append(curr_dct['frameExamples'][0])
        list_info_dct['listCoreRoles'].append(curr_dct['roles'])
        list_info_dct['listRoleDefinitions'].append(curr_dct['roleDefinitions'])
        list_info_dct['listRoleExamples'].append(curr_dct['roleExamples'])

    return list_info_dct

def frame_to_directly_related_frames(frame: str):
    """
    Given a frame, find all its direct children or directly related frames
    """
    children = set()

    for idx, relation in enumerate(fn.frame_relations(frame)):
        if relation['type'].name == 'Inheritance' or relation['type'].name == 'Using' :
            child_frame = relation["Child"].name
            children.add(child_frame)
            #parent_frame = relation["Parent"].name
            #children.add(parent_frame)

        elif relation['type'].name == 'Subframe':
            component_frame = relation["Component"].name
            children.add(component_frame)
            # complex_frame = relation["Complex"].name
            # children.add(complex_frame)
            
        elif relation['type'].name == 'Precedes':
            earlier_frame = relation["Earlier"].name
            later_frame = relation["Later"].name
            children.add(earlier_frame)
            children.add(later_frame)

        elif relation['type'].name == 'Perspective_on':
            neutral_frame = relation["Neutral"].name
            perspectivized_frame = relation["Perspectivized"].name
            children.add(neutral_frame)
            children.add(perspectivized_frame)

        else:
            continue

    return children


def frame_to_children(frame: str):
    """
    Given a frame, find all its direct children (either inheritance or subframe)
    """
    children = set()

    for idx, relation in enumerate(fn.frame_relations(frame)):
        if relation['type'].name == 'Inheritance':
            child_frame = relation["Child"].name
            children.add(child_frame)
            #parent_frame = relation["Parent"].name
            #children.add(parent_frame)

        elif relation['type'].name == 'Subframe':
            component_frame = relation["Component"].name
            children.add(component_frame)
            # complex_frame = relation["Complex"].name
            # children.add(complex_frame)
        else:
            continue

    return children


def bfs_frame(frame: str,
               descendants_only: bool = False):
    """
    Given a frame, find all its descendants using BFS
    """
    children = set()
    queue = [frame]

    if descendants_only:
        while queue:
            current_frame = queue.pop(0)
            children.add(current_frame)
            # print(f"##### Children of {current_frame} ####")
            for child in frame_to_children(current_frame):
                if child not in children:
                    # print(f"{child}")
                    queue.append(child)
    else:
        while queue:
            current_frame = queue.pop(0)
            children.add(current_frame)
            # print(f"##### Children of {current_frame} ####")
            for child in frame_to_directly_related_frames(current_frame):
                if child not in children:
                    # print(f"{child}")
                    queue.append(child)
    return children     

def get_all_relations(frame, relation_type='Inheritance', variable = 'Child'):
    """
    Returns all the children of a frame in the FrameNet hierarchy
    Includes duplicates, so you may want to use set() on the output
    Note: This function works with all relation types.
    Inheritance: Parent/Child
    Subframe: Complex/Component
    Precedes: Earlier/Later
    """
    children = []
    for relation in fn.frame(frame).frameRelations:
        if relation.type.name == relation_type:
            if relation.get(variable).name != frame:
                children.append(relation.get(variable).name)
                children.extend(get_all_relations(relation.get(variable).name, relation_type=relation_type, variable=variable))
    return children



# # all_event_frames = bfs_children("Event")
# ALL_FRAMES = set([frame.name for frame in fn.frames()])
# # all_event_frames = bfs_children("Event")
# entity_frames = bfs_frame("Entity", descendants_only=True)
# locale_frames = bfs_frame("Locale", descendants_only=True)
# event_frames = bfs_frame("Event", descendants_only=True)
# state_frames = bfs_frame("State", descendants_only=True)
# process_frames = bfs_frame("Process", descendants_only=True)

# SITUATION_FRAMES = event_frames.union(state_frames, process_frames)